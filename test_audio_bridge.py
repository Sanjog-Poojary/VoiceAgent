import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from audio_bridge import AudioBridge
from orchestrator import VoiceAgentWorkflow, TurnClassification
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.apps import App, ResumabilityConfig
import copy

class MockTwilioWS:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def send_text(self, text):
        self.sent_messages.append(json.loads(text))

    async def close(self):
        self.closed = True

class TestAudioBridge(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Set up a real local ADK runner and session service
        self.session_service = InMemorySessionService()
        self.workflow = VoiceAgentWorkflow(name="voice_agent_workflow")
        self.adk_app = App(
            name="VoiceAgent",
            root_agent=self.workflow,
            resumability_config=ResumabilityConfig(is_resumable=True)
        )
        self.runner = Runner(
            app=self.adk_app,
            session_service=self.session_service,
            auto_create_session=True
        )

        self.user_id = "test_user"
        self.session_id = "test_session"
        self.mock_twilio = MockTwilioWS()

        # Create session in mock service
        await self.session_service.create_session(
            app_name="VoiceAgent",
            user_id=self.user_id,
            session_id=self.session_id,
            state={
                "customer_id": "1",
                "detected_language": "English",
                "current_agent": "IdentityAgent",
                "verification_attempts": 0,
                "call_sentiment": "Neutral",
                "offer_pitched": False,
                "offer_accepted": False,
                "escalation_triggered": False,
                "raw_audio_transcription": []
            }
        )

        # Create the AudioBridge instance in dummy mode
        self.bridge = AudioBridge(
            twilio_ws=self.mock_twilio,
            runner=self.runner,
            session_service=self.session_service,
            user_id=self.user_id,
            session_id=self.session_id,
            deepgram_api_key="dummy_key"
        )
        self.bridge.stream_sid = "mock_sid_123"

    async def asyncTearDown(self):
        await self.bridge.close()

    @patch("orchestrator.fetch_customer_details")
    @patch("orchestrator.fetch_all_offers")
    @patch("orchestrator.classify_turn")
    @patch("orchestrator.fetch_event_triggers")
    async def test_successful_reasoning_and_commit(self, mock_event, mock_classify, mock_offers, mock_customer):
        mock_event.return_value = {"id": "ev_1", "customer_id": "1", "event_type": "Birthday", "event_date": "2026-06-29"}
        mock_customer.return_value = {
            "id": "1", "name": "Sanjog", "phone": "+1234567890",
            "base_language": "English", "preferred_category": "Fashion",
            "secondary_brand": "Puma"
        }
        mock_offers.return_value = [
            {
                "offer_id": "off_1",
                "offer_name": "BIRTHDAY20",
                "offer_brand": "Stop",
                "offer_category": "Fashion",
                "valid_from": "2026-06-29",
                "valid_to": "2026-07-29",
                "offer_description": "Get 20% off on Stop everyday casuals."
            }
        ]
        # Mock classify_turn to return a successful identity confirmation
        mock_classify.return_value = TurnClassification(
            detected_language="English",
            call_sentiment="Neutral",
            is_valid_answer=True,
            is_acceptance=False,
            is_decline=False,
            is_third_party=False,
            is_competitor_mention=False,
            is_loyalty_question=False,
            is_appointment_accept=False,
            is_appointment_decline=False,
            is_injection_attempt=False,
            preferred_slot="",
            is_silent_turn=False,
            is_knowledge_question=False,
            knowledge_query="",
            ambiguity_reason="",
            confidence_score=0.95
        )

        # 1. Start the reasoning task
        self.bridge.reasoning_task = asyncio.create_task(self.bridge.task_reasoning_adk())

        # 2. Push initial turn to greeting
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 0,
            "text": "[Call Connected]"
        })
        await asyncio.sleep(0.5)

        # Flush initial greeting from tts queue
        self.assertEqual(self.bridge.outbound_tts_queue.qsize(), 1)
        greet_msg = await self.bridge.outbound_tts_queue.get()
        self.assertIn("speaking with Sanjog", greet_msg["text"])

        # 3. Push user confirmation turn (turn_id = 1)
        self.bridge.current_turn_id = 1
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 1,
            "text": "Yes, I am Sanjog."
        })

        # Allow execution loop to run
        await asyncio.sleep(0.5)

        # Check that output text was produced and pushed to tts queue
        self.assertEqual(self.bridge.outbound_tts_queue.qsize(), 1)
        tts_item = await self.bridge.outbound_tts_queue.get()
        self.assertEqual(tts_item["turn_id"], 1)
        self.assertIn("Happy Birthday", tts_item["text"])

        # State should be updated (next agent should be SalesPitchAgent)
        session = await self.session_service.get_session(
            app_name="VoiceAgent",
            user_id=self.user_id,
            session_id=self.session_id
        )
        self.assertEqual(session.state.get("current_agent"), "SalesPitchAgent")

    @patch("orchestrator.fetch_customer_details")
    @patch("orchestrator.fetch_all_offers")
    @patch("orchestrator.classify_turn")
    @patch("orchestrator.fetch_event_triggers")
    @patch("orchestrator.send_email_notification")
    async def test_email_notification_preference(self, mock_send_email, mock_event, mock_classify, mock_offers, mock_customer):
        mock_send_email.return_value = {"status": "success"}
        mock_event.return_value = {"id": "ev_1", "customer_id": "1", "event_type": "Birthday", "event_date": "2026-06-29"}
        mock_customer.return_value = {
            "id": "1", "name": "Sanjog", "phone": "+1234567890", "email": "sanjog@example.com",
            "base_language": "English", "preferred_category": "Fashion",
            "secondary_brand": "Puma"
        }
        mock_offers.return_value = [
            {
                "offer_id": "off_1",
                "offer_name": "BIRTHDAY20",
                "offer_brand": "Stop",
                "offer_category": "Fashion",
                "valid_from": "2026-06-29",
                "valid_to": "2026-07-29",
                "offer_description": "Get 20% off on Stop everyday casuals."
            }
        ]

        self.bridge.reasoning_task = asyncio.create_task(self.bridge.task_reasoning_adk())

        # 1. Greeting
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 0,
            "text": "[Call Connected]"
        })
        await asyncio.sleep(0.5)
        await self.bridge.outbound_tts_queue.get()

        # 2. Confirmation (user wants email instead of WhatsApp)
        mock_classify.return_value = TurnClassification(
            detected_language="English", call_sentiment="Neutral",
            is_valid_answer=True, is_acceptance=True, is_decline=False,
            is_third_party=False, is_competitor_mention=False, is_loyalty_question=False,
            is_appointment_accept=False, is_appointment_decline=False, is_injection_attempt=False,
            preferred_slot="", is_silent_turn=False, is_knowledge_question=False,
            knowledge_query="", ambiguity_reason="", confidence_score=0.95
        )
        self.bridge.current_turn_id = 1
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 1,
            "text": "send it to my email"
        })
        await asyncio.sleep(0.5)
        
        # Flush the secondary Puma pitch (since they have a secondary brand)
        await self.bridge.outbound_tts_queue.get()

        # 3. User accepts Puma offer
        self.bridge.current_turn_id = 2
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 2,
            "text": "ok"
        })
        await asyncio.sleep(0.5)

        # Check final TTS output
        self.assertEqual(self.bridge.outbound_tts_queue.qsize(), 1)
        final_msg = await self.bridge.outbound_tts_queue.get()
        self.assertIn("directly to your email", final_msg["text"])
        mock_send_email.assert_called_once()

    @patch("orchestrator.fetch_customer_details")
    @patch("orchestrator.fetch_all_offers")
    @patch("orchestrator.classify_turn")
    @patch("orchestrator.fetch_event_triggers")
    async def test_barge_in_rollbacks_state(self, mock_event, mock_classify, mock_offers, mock_customer):
        mock_event.return_value = {"id": "ev_1", "customer_id": "1", "event_type": "Birthday", "event_date": "2026-06-29"}
        mock_customer.return_value = {
            "id": "1", "name": "Sanjog", "phone": "+1234567890",
            "base_language": "English", "preferred_category": "Fashion",
            "secondary_brand": "Puma"
        }
        mock_offers.return_value = [
            {
                "offer_id": "off_1",
                "offer_name": "BIRTHDAY20",
                "offer_brand": "Stop",
                "offer_category": "Fashion",
                "valid_from": "2026-06-29",
                "valid_to": "2026-07-29",
                "offer_description": "Get 20% off on Stop everyday casuals."
            }
        ]
        mock_classify.return_value = TurnClassification(
            detected_language="English",
            call_sentiment="Neutral",
            is_valid_answer=True,
            is_acceptance=False,
            is_decline=False,
            is_third_party=False,
            is_competitor_mention=False,
            is_loyalty_question=False,
            is_appointment_accept=False,
            is_appointment_decline=False,
            is_injection_attempt=False,
            preferred_slot="",
            is_silent_turn=False,
            is_knowledge_question=False,
            knowledge_query="",
            ambiguity_reason="",
            confidence_score=0.95
        )

        # 1. Start reasoning
        self.bridge.reasoning_task = asyncio.create_task(self.bridge.task_reasoning_adk())

        # 2. Push initial turn to greeting
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 0,
            "text": "[Call Connected]"
        })
        await asyncio.sleep(0.5)
        await self.bridge.outbound_tts_queue.get()

        # 3. Push user confirmation turn (turn_id = 1)
        self.bridge.current_turn_id = 1
        await self.bridge.turn_queue.put({
            "session_id": self.session_id,
            "turn_id": 1,
            "text": "Yes, it is."
        })

        # 4. Simulate immediate user speech barge-in (SpeechStarted increments turn_id to 2)
        self.bridge.current_turn_id = 2

        # Allow turn 1 execution to finish
        await asyncio.sleep(0.5)

        # Assert output was discarded (nothing in tts queue)
        self.assertEqual(self.bridge.outbound_tts_queue.qsize(), 0)

        # Assert state was rolled back (current_agent remains IdentityAgent because turn 1 was discarded)
        session = await self.session_service.get_session(
            app_name="VoiceAgent",
            user_id=self.user_id,
            session_id=self.session_id
        )
        self.assertEqual(session.state.get("current_agent"), "IdentityAgent")

    async def test_silence_timer_expiry(self):
        # 1. Start silence timer
        self.bridge.silence_timer_task = asyncio.create_task(self.bridge.run_silence_timer(0))

        # 2. Await expiry (waits 4.5s; let's mock sleep or wait for it)
        await asyncio.sleep(4.7)

        # 3. Check turn_queue contains the synthetic silence turn "..."
        self.assertEqual(self.bridge.turn_queue.qsize(), 1)
        item = await self.bridge.turn_queue.get()
        self.assertEqual(item["text"], "...")
        self.assertEqual(item["turn_id"], 0)

    async def test_silence_timer_cancelled_by_new_speech(self):
        # 1. Start silence timer
        self.bridge.silence_timer_task = asyncio.create_task(self.bridge.run_silence_timer(0))
        await asyncio.sleep(2.0)

        # 2. SpeechStarted triggers and increments turn_id (cancelling timer task)
        self.bridge.current_turn_id = 1
        if self.bridge.silence_timer_task and not self.bridge.silence_timer_task.done():
            self.bridge.silence_timer_task.cancel()

        await asyncio.sleep(3.0)

        # 3. Queue should remain empty because it was cancelled
        self.assertEqual(self.bridge.turn_queue.qsize(), 0)

    def test_software_squelch(self):
        self.bridge.active_tts_string = "Would you like me to send these details to your WhatsApp?"
        
        # Test Case A: Echo (high similarity) -> Squelched (True)
        self.assertTrue(self.bridge.software_squelch("send these details to your WhatsApp", 0.95))

        # Test Case B: Distinct user speech, high confidence -> Accepted (False)
        self.assertFalse(self.bridge.software_squelch("Wait, who is this calling?", 0.90))

        # Test Case C: Low confidence -> Squelched (True)
        self.assertTrue(self.bridge.software_squelch("no please", 0.50))
