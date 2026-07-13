import asyncio
import base64
import copy
import difflib
import json
import logging
import os
import time
from typing import Optional
import websockets
from google.genai import types

logger = logging.getLogger("audio_bridge")

def make_user_message(text: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[types.Part(text=text)]
    )

def make_resume_message(interrupt_id: str, text: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=interrupt_id,
                    name="adk_request_input",
                    response={"result": text}
                )
            )
        ]
    )

def get_agent_message_text(event) -> str:
    if event.author in ("orchestrator_llm", "orchestrator"):
        return ""
    if event.output:
        if isinstance(event.output, str):
            return event.output
        if isinstance(event.output, dict):
            trans = event.output.get("raw_audio_transcription", [])
            if trans and isinstance(trans[-1], str) and trans[-1].startswith("Agent: "):
                return trans[-1][7:]
    if not event.content or not event.content.parts:
        return ""
    for part in event.content.parts:
        if part.text:
            return part.text
        if part.function_call and part.function_call.name == "adk_request_input":
            return part.function_call.args.get("message", "")
    return ""

def get_interrupt_id(event) -> str | None:
    if not event.content or not event.content.parts:
        return None
    for part in event.content.parts:
        if part.function_call and part.function_call.name == "adk_request_input":
            return part.function_call.id
    return None

class AudioBridge:
    def __init__(
        self,
        twilio_ws,
        runner,
        session_service,
        user_id: str,
        session_id: str,
        deepgram_api_key: Optional[str] = None
    ):
        self.twilio_ws = twilio_ws
        self.runner = runner
        self.session_service = session_service
        self.user_id = user_id
        self.session_id = session_id
        self.deepgram_api_key = deepgram_api_key or os.getenv("DEEPGRAM_API_KEY", "dummy_key")

        # Queues
        self.inbound_audio_queue = asyncio.Queue()
        self.turn_queue = asyncio.Queue()
        self.outbound_tts_queue = asyncio.Queue()

        # Shared synchronization
        self.session_lock = asyncio.Lock()
        self.filler_lock = asyncio.Lock()

        # State tracking
        self.current_turn_id = 0
        self.is_speaking = False
        self.active_tts_string = ""
        self.stream_sid = None
        self.call_sid = None
        self.last_interrupt_id = None
        self.last_invocation_id = None

        # Task references
        self.stt_task = None
        self.reasoning_task = None
        self.tts_task = None
        self.silence_timer_task = None

    def software_squelch(self, transcript: str, confidence: float) -> bool:
        """
        Compare the transcript against active_tts_string (echo suppression).
        """
        if not self.active_tts_string:
            return False

        similarity = difflib.SequenceMatcher(None, transcript.lower(), self.active_tts_string.lower()).ratio()
        if similarity > 0.65:
            logger.info(f"Squelched (echo detected): {transcript!r} similarity {similarity:.2f}")
            return True

        if confidence < 0.70:
            logger.info(f"Squelched (low confidence {confidence:.2f}): {transcript!r}")
            return True

        return False

    async def get_tts_audio(self, text: str) -> bytes:
        if not self.deepgram_api_key or self.deepgram_api_key.startswith("dummy"):
            # Dummy mode: return mocked mulaw sound chunks (0xff bytes) proportional to string length
            return b"\xff" * (len(text) * 80)

        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepgram.com/v1/speak?model=aura-2-asteria-en&encoding=mulaw&sample_rate=8000",
                headers={
                    "Authorization": f"Token {self.deepgram_api_key}",
                    "Content-Type": "application/json"
                },
                json={"text": text},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"TTS API error status {response.status_code}: {response.text}")
                return b""

    async def trigger_filler_timer(self):
        try:
            # Spawns a background timer: if execution exceeds 1.5s, trigger filler audio
            await asyncio.sleep(1.5)
            logger.info("Reasoning exceeded 1.5s; playing filler audio.")
            
            # Bound lock acquisition to 5.0 seconds (covers classify_turn retry + LLM latency)
            try:
                await asyncio.wait_for(self.filler_lock.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout acquiring filler_lock for filler audio")
                return

            try:
                # 2.0 seconds of mulaw silence/filler (8000 samples/sec = 16000 bytes)
                dummy_filler = b"\xff" * 16000
                chunk_size = 640  # 80ms
                for i in range(0, len(dummy_filler), chunk_size):
                    if not self.stream_sid:
                        break
                    chunk = dummy_filler[i:i+chunk_size]
                    payload = base64.b64encode(chunk).decode("utf-8")
                    await self.twilio_ws.send_json({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": payload}
                    })
                    await asyncio.sleep(0.07)
            finally:
                if self.filler_lock.locked():
                    self.filler_lock.release()
        except asyncio.CancelledError:
            pass

    async def run_silence_timer(self, turn_id: int):
        try:
            await asyncio.sleep(4.5)
            logger.info(f"Silence timer expired for turn {turn_id}. Triggering synthetic turn.")
            if turn_id == self.current_turn_id:
                await self.turn_queue.put({
                    "session_id": self.session_id,
                    "turn_id": turn_id,
                    "text": "..."
                })
        except asyncio.CancelledError:
            pass

    async def task_inbound_stt(self):
        """
        Receives raw audio from inbound_audio_queue, connects to Deepgram STT,
        handles SpeechStarted (clearing Twilio buffer), and pushes final text.
        """
        url = "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=mulaw&sample_rate=8000&channels=1&interim_results=true&vad_events=true&endpointing=500"
        headers = {"Authorization": f"Token {self.deepgram_api_key}"}

        # Handle offline/mock testing mode
        if self.deepgram_api_key.startswith("dummy"):
            logger.info("Running STT task in mock/dummy mode.")
            # Run offline loop just waiting for queue
            while True:
                data = await self.inbound_audio_queue.get()
                # Mock STT does not process audio bytes directly
                await asyncio.sleep(0.01)
            return

        try:
            async with websockets.connect(url, additional_headers=headers) as dg_ws:
                # Helper to send inbound queue audio to Deepgram
                async def send_audio():
                    while True:
                        audio = await self.inbound_audio_queue.get()
                        await dg_ws.send(audio)

                send_task = asyncio.create_task(send_audio())

                try:
                    async for message in dg_ws:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "SpeechStarted":
                            # Cancel silence timer immediately
                            if self.silence_timer_task and not self.silence_timer_task.done():
                                self.silence_timer_task.cancel()
                            logger.info("SpeechStarted event received. Cancelled silence timer.")

                        elif msg_type == "Results":
                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)
                            if is_final or speech_final:
                                alternatives = data.get("channel", {}).get("alternatives", [])
                                if alternatives:
                                    transcript = alternatives[0].get("transcript", "").strip()
                                    confidence = alternatives[0].get("confidence", 0.0)
                                    if transcript:
                                        logger.info(f"STT Transcript: {transcript!r} (confidence {confidence:.2f})")
                                        if not self.software_squelch(transcript, confidence):
                                            # If agent is actively speaking, this is a valid user barge-in!
                                            if self.is_speaking:
                                                self.current_turn_id += 1
                                                logger.info(f"Barge-in detected via valid transcript. Incremented current_turn_id to {self.current_turn_id}")
                                                
                                                # Send Clear to Twilio to stop playback immediately
                                                if self.stream_sid:
                                                    await self.twilio_ws.send_json({
                                                        "event": "clear",
                                                        "streamSid": self.stream_sid
                                                    })
                                                self.is_speaking = False
                                            
                                            # Push transcript to reasoning queue
                                            await self.turn_queue.put({
                                                "session_id": self.session_id,
                                                "turn_id": self.current_turn_id,
                                                "text": transcript
                                            })
                finally:
                    send_task.cancel()
        except Exception as e:
            logger.error(f"Deepgram STT connection error: {e}", exc_info=True)

    async def task_reasoning_adk(self):
        """
        Pops turn from queue, serializes per-session via Lock, snapshots state,
        runs ADK workflow, triggers 1.5s filler, commits or restores state on barge-in.
        """
        while True:
            turn = await self.turn_queue.get()
            session_id = turn["session_id"]
            turn_id = turn["turn_id"]
            text = turn["text"]

            async with self.session_lock:
                session = await self.session_service.get_session(
                    app_name="VoiceAgent",
                    user_id=self.user_id,
                    session_id=session_id
                )
                if not session:
                    continue

                # Snapshot the state deeply
                snapshot_state = copy.deepcopy(session.state)

                filler_timer = asyncio.create_task(self.trigger_filler_timer())

                agent_message = ""
                try:
                    if self.last_interrupt_id:
                        new_msg = make_resume_message(self.last_interrupt_id, text)
                    else:
                        new_msg = make_user_message(text)

                    async for event in self.runner.run_async(
                        user_id=self.user_id,
                        session_id=session_id,
                        new_message=new_msg,
                        invocation_id=self.last_invocation_id
                    ):
                        if event.invocation_id:
                            self.last_invocation_id = event.invocation_id
                        iid = get_interrupt_id(event)
                        if iid:
                            self.last_interrupt_id = iid
                        msg = get_agent_message_text(event)
                        if msg:
                            agent_message = msg
                except Exception as e:
                    logger.error(f"Error in reasoning ADK run: {e}", exc_info=True)
                finally:
                    filler_timer.cancel()

                # Handle rollbacks if barge-in occurred
                if turn_id == self.current_turn_id:
                    # Match: commit (state modifications are live in session_service)
                    logger.info(f"Turn {turn_id} reasoning finished. Yielding: {agent_message!r}")
                    if agent_message:
                        await self.outbound_tts_queue.put({
                            "text": agent_message,
                            "turn_id": turn_id
                        })
                else:
                    # Mismatch: overwrite memory state back to pre-run snapshot
                    logger.info(f"Barge-in detected (turn {turn_id} vs current {self.current_turn_id}). Rolling back state.")
                    storage_session = self.session_service.sessions["VoiceAgent"][self.user_id].get(session_id)
                    if storage_session:
                        storage_session.state = copy.deepcopy(snapshot_state)

    async def task_outbound_tts(self):
        """
        Pops from outbound_tts_queue, calls Deepgram Aura, awaits release
        of filler lock, streams mulaw chunks, starts silence timer.
        """
        while True:
            item = await self.outbound_tts_queue.get()
            text = item["text"]
            turn_id = item["turn_id"]

            if turn_id != self.current_turn_id:
                continue

            try:
                audio_bytes = await self.get_tts_audio(text)
            except Exception as e:
                logger.error(f"Error generating TTS: {e}")
                audio_bytes = b""

            # Await release of filler lock if held
            async with self.filler_lock:
                if turn_id != self.current_turn_id:
                    continue

                self.is_speaking = True
                self.active_tts_string = text

                # Stream to Twilio in 640 byte (80ms) chunks
                chunk_size = 640
                for i in range(0, len(audio_bytes), chunk_size):
                    if turn_id != self.current_turn_id:
                        break
                    chunk = audio_bytes[i:i+chunk_size]
                    payload = base64.b64encode(chunk).decode("utf-8")
                    if self.stream_sid:
                        await self.twilio_ws.send_json({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": payload}
                        })
                    # Sleep slightly less than 80ms (e.g. 70ms) to ensure Twilio's buffer is never starved
                    await asyncio.sleep(0.07)

                self.is_speaking = False
                self.active_tts_string = ""

                # Start/restart silence timer
                if turn_id == self.current_turn_id:
                    if self.silence_timer_task and not self.silence_timer_task.done():
                        self.silence_timer_task.cancel()
                    self.silence_timer_task = asyncio.create_task(self.run_silence_timer(turn_id))

                # If the next agent is Terminate, hang up the call after speaking!
                if turn_id == self.current_turn_id:
                    session = await self.session_service.get_session(
                        app_name="VoiceAgent",
                        user_id=self.user_id,
                        session_id=self.session_id
                    )
                    if session and session.state.get("current_agent") == "Terminate":
                        logger.info("Termination agent reached. Hanging up Twilio call.")
                        # Wait a brief moment for Twilio buffer to play the goodbye audio fully
                        await asyncio.sleep(1.0)
                        await self.hangup_twilio_call()

    async def start(self):
        self.stt_task = asyncio.create_task(self.task_inbound_stt())
        self.reasoning_task = asyncio.create_task(self.task_reasoning_adk())
        self.tts_task = asyncio.create_task(self.task_outbound_tts())

    async def close(self):
        for t in (self.stt_task, self.reasoning_task, self.tts_task, self.silence_timer_task):
            if t and not t.done():
                t.cancel()

    async def hangup_twilio_call(self):
        if not self.call_sid or not self.twilio_ws:
            return
            
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token or account_sid.startswith("dummy"):
            logger.warning("Twilio credentials missing. Closing websocket only.")
            try:
                await self.twilio_ws.close()
            except Exception:
                pass
            return
            
        import httpx
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{self.call_sid}.json"
        try:
            # We must use Basic Auth for Twilio REST API
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    auth=(account_sid, auth_token),
                    data={"Status": "completed"},
                    timeout=5.0
                )
                if resp.status_code == 200:
                    logger.info(f"Successfully hung up Twilio call {self.call_sid}")
                else:
                    logger.error(f"Failed to hang up Twilio call: {resp.text}")
        except Exception as e:
            logger.error(f"Error hanging up Twilio call: {e}")
        finally:
            try:
                await self.twilio_ws.close()
            except Exception:
                pass
