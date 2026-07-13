import os
import unittest
import asyncio
import httpx
from pydantic import BaseModel
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.apps import App, ResumabilityConfig
from google.genai import types

try:
    from orchestrator import VoiceAgentWorkflow
except ModuleNotFoundError:
    from VoiceAgent.orchestrator import VoiceAgentWorkflow

MOCK_SERVER_URL = os.getenv("MOCK_SERVER_URL", "http://127.0.0.1:8000")
INTER_TURN_SLEEP = 4.0

def make_initial_state(customer_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "current_agent": "IdentityAgent",
        "detected_language": "English",
        "verification_attempts": 0,
        "offer_pitched": False,
        "offer_accepted": False,
        "raw_audio_transcription": []
    }

def make_user_message(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])

def make_resume_message(interrupt_id: str, text: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(
            id=interrupt_id, name="adk_request_input", response={"result": text}
        ))]
    )

def get_agent_message_text(event) -> str:
    if event.author in ("orchestrator_llm", "orchestrator", "orchestrator_node"):
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

async def run_turn(runner, user_id, session_id, message, invocation_id=None, state_delta=None):
    kwargs = dict(user_id=user_id, session_id=session_id, new_message=message, invocation_id=invocation_id)
    if state_delta is not None:
        kwargs["state_delta"] = state_delta

    agent_message = ""
    next_interrupt_id = ""
    next_invocation_id = invocation_id

    async for event in runner.run_async(**kwargs):
        if event.invocation_id:
            next_invocation_id = event.invocation_id
        iid = get_interrupt_id(event)
        if iid:
            next_interrupt_id = iid
        msg = get_agent_message_text(event)
        if msg:
            agent_message = msg

    return agent_message, next_interrupt_id, next_invocation_id


class TestVoiceAgentOrchestrator(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
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
        self.user_id = "test_user_id"

    async def get_session_state(self, session_id: str):
        session = await self.session_service.get_session(app_name="VoiceAgent", user_id=self.user_id, session_id=session_id)
        return session.state if session else {}

    def make_initial_state(self, customer_id: str) -> dict:
        return make_initial_state(customer_id)

    # SCENARIO A: English Happy Path
    async def test_scenario_a_happy_path(self):
        """
        Scenario A: Customer connects, confirms identity, receives birthday offer, accepts,
        receives secondary brand upsell, accepts both.
        """
        session_id = "happy_path_session"
        print("\n=======================================================")
        print("SCENARIO A: English Happy Path (Collapsed Pitch)")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )
        self.assertIn("speaking with Sanjog", agent_message)
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "IdentityAgent")

        print(f"\n--- Turn 2: User confirms identity ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes, this is Sanjog"), invocation_id
        )
        self.assertIn("BIRTHDAY20", agent_message)
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "SalesPitchAgent")
        self.assertTrue(state.get("offer_pitched"))

        print(f"\n--- Turn 3: User accepts primary offer ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes please, send it to my WhatsApp"), invocation_id
        )
        self.assertIn("Puma", agent_message)
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "SalesPitchAgent")

        print(f"\n--- Turn 4: User accepts secondary offer ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes please, include that as well"), invocation_id
        )
        self.assertIn("WhatsApp", agent_message)
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "PostCallAgent")
        self.assertTrue(state.get("offer_accepted"))

        print(f"\n--- Turn 5: User says bye ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Thank you, bye"), invocation_id
        )
        self.assertEqual(agent_message, "Goodbye!")
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "Terminate")

    # SCENARIO B: Hindi Escalation
    async def test_scenario_b_escalation_hindi(self):
        """
        Scenario B: Customer confirms identity in Hindi, agent pitches birthday offer in Hindi,
        customer gets angry/escalates, routed to Apology/Escalation.
        """
        session_id = "escalation_hindi_session"
        print("\n=======================================================")
        print("SCENARIO B: Hindi Escalation")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("2")
        )
        self.assertIn("speaking with Aarav", agent_message)

        print(f"\n--- Turn 2: User confirms in Hindi ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "हाँ, मैं आरव बात कर रहा हूँ।"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("detected_language"), "Hindi")
        self.assertIn("फर्स्ट सिटीजन", agent_message)

        print(f"\n--- Turn 3: User escalates/gets angry ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "बिल्कुल बकवास सर्विस है तुम्हारी, मुझे कोई बात नहीं करनी!"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertIn(state.get("current_agent"), ("ApologyAgent", "EscalationAgent", "Terminate"))
        print(f"\n[PASS] Handled escalation successfully. Agent: {state.get('current_agent')}")

    # SCENARIO C: Suspicious Gatekeeper (Third Party)
    async def test_scenario_c_suspicious_gatekeeper(self):
        """
        Scenario C: Caller is a relative/gatekeeper, not the target.
        ASSERT: Guardrails trigger, routes directly to PostCallAgent / terminates.
        """
        session_id = "gatekeeper_session"
        print("\n=======================================================")
        print("SCENARIO C: The Suspicious Gatekeeper (Third Party)")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Gatekeeper answers ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "No, he's not available. I am his husband speaking."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "ApologyAgent")
        self.assertFalse(state.get("offer_pitched"), "FAIL: Offer was pitched to a third party.")
        print(f"\n[PASS] Third party gatekeeper handled safely. Agent: {state.get('current_agent')}")

    # SCENARIO D: Ambiguous Identity (Loop exit)
    async def test_scenario_d_ambiguous_identity(self):
        """
        Scenario D: Customer never gives a clear Yes/No to identity checks.
        ASSERT: System exits verification loop by turn 4.
        """
        session_id = "ambiguous_identity_session"
        print("\n=======================================================")
        print("SCENARIO D: The Ambiguous Identity")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Ambiguous reply 1 ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Who is asking? Depends on who wants to know."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "IdentityAgent")

        print(f"\n--- Turn 3: Ambiguous reply 2 ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Maybe, depends. Why are you calling?"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "IdentityAgent")

        print(f"\n--- Turn 4: Ambiguous reply 3 (loop must exit) ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "I really cannot say right now."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertNotEqual(state.get("current_agent"), "IdentityAgent")
        self.assertIn(state.get("current_agent"), ("ApologyAgent", "Terminate"))
        print(f"\n[PASS] Loop exited. Final agent: {state.get('current_agent')}")

    # SCENARIO E: Mid-Call Language Switch
    async def test_scenario_e_mid_call_language_switch(self):
        """
        Scenario E: Customer starts in English, switches to Hindi after SalesPitchAgent pitch.
        ASSERT: detected_language updates to Hindi and offer response is in Hindi.
        """
        session_id = "code_switch_session"
        print("\n=======================================================")
        print("SCENARIO E: Mid-Call Code Switcher (Hinglish)")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Confirms in English ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes, it is me, Sanjog."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "SalesPitchAgent")

        print(f"\n--- Turn 3: Switches to Hindi mid-call ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Arre yaar, ab Hindi mein baat karo. Offer kya hai dobara batao?"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("detected_language"), "Hindi")
        self.assertIn("ऑफ़र", agent_message)
        print(f"\n[PASS] Language switched mid-call. Agent: {state.get('current_agent')}")

    # SCENARIO F: Internet Slang
    async def test_scenario_f_internet_slang(self):
        """
        Scenario F: Customer replies in internet slang.
        ASSERT: Does not bypass identity verification.
        """
        session_id = "slang_session"
        print("\n=======================================================")
        print("SCENARIO F: Internet Slang / Unsupported Dialect")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Internet slang response ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "no cap fr fr skibidi rizz"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertNotEqual(state.get("current_agent"), "SalesPitchAgent")
        self.assertFalse(state.get("offer_pitched"))
        print(f"\n[PASS] Slang handled safely. Agent: {state.get('current_agent')}")

    # SCENARIO G: Competitor Baiter
    async def test_scenario_g_competitor_baiter(self):
        """
        Scenario G: Customer mentions competitor Zara/Lifestyle.
        ASSERT: Safety guardrail routes out.
        """
        session_id = "competitor_baiter_session"
        print("\n=======================================================")
        print("SCENARIO G: The Competitor Baiter")
        print("=======================================================")

        # Start call
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        # Confirm identity to pitch offer
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes, this is Sanjog"), invocation_id
        )

        print(f"\n--- Turn 3: User mentions competitor ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Can I redeem this coupon at Zara or Lifestyle instead?"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertNotEqual(state.get("current_agent"), "SalesPitchAgent")
        print(f"\n[PASS] Competitor brand baiting blocked. Agent: {state.get('current_agent')}")

    # SCENARIO H: Silence Recovery
    async def test_scenario_h_silence_recovery(self):
        """
        Scenario H: Customer is silent on verification.
        ASSERT: Re-prompts up to 3 times before disconnecting.
        """
        session_id = "silence_session"
        print("\n=======================================================")
        print("SCENARIO H: Silence Recovery")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Silent user ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "..."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "IdentityAgent")
        print(f"\n[PASS] Reprompts on silence. Agent: {state.get('current_agent')}")

    # SCENARIO I: Sarcastic Spender
    async def test_scenario_i_sarcastic_spender(self):
        """
        Scenario I: Sarcastic statement.
        ASSERT: Agitated sentiment detected.
        """
        session_id = "sarcastic_session"
        print("\n=======================================================")
        print("SCENARIO I: The Sarcastic Spender")
        print("=======================================================")

        # Connect
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("3")
        )

        # Confirm identity
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes, I am Aarav"), invocation_id
        )

        print(f"\n--- Turn 3: Sarcastic response ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Oh wow, my credits are expiring. AMAZING. How wonderful of you to let me know."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("call_sentiment"), "Agitated")
        print(f"\n[PASS] Sarcasm classified as Agitated. Sentiment: {state.get('call_sentiment')}")

    # SCENARIO J: Silent User (Max Silent Turns)
    async def test_scenario_j_silent_user(self):
        """
        Scenario J: 3 turns of silence.
        ASSERT: Ends call.
        """
        session_id = "max_silence_session"
        print("\n=======================================================")
        print("SCENARIO J: The Silent User (Max Silent Turns)")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Silent turn 1 ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "..."), invocation_id
        )

        print(f"\n--- Turn 3: Silent turn 2 ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "..."), invocation_id
        )

        print(f"\n--- Turn 4: Silent turn 3 (should disconnect) ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "..."), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "Terminate")
        print(f"\n[PASS] Disconnects after 3 silences. Agent: {state.get('current_agent')}")

    # SCENARIO K: Context Breaker
    async def test_scenario_k_context_breaker(self):
        """
        Scenario K: After OfferAgent pitches, customer asks about loyalty points.
        ASSERT: Satisfies tangent, then returns to pitch.
        """
        session_id = "context_breaker_session"
        print("\n=======================================================")
        print("SCENARIO K: The Context Breaker")
        print("=======================================================")

        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_user_message("[Call Connected]"), state_delta=self.make_initial_state("1")
        )

        print(f"\n--- Turn 2: Confirm identity ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes, this is Sanjog"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "SalesPitchAgent")
        self.assertTrue(state.get("offer_pitched"))

        print(f"\n--- Turn 3: Context break - asks about loyalty points ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Wait, before I decide - what is my loyalty tier? How many points do I have?"), invocation_id
        )
        self.assertIn("Gold Tier", agent_message)

        print(f"\n--- Turn 4: User accepts offer ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, interrupt_id, invocation_id = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Okay yes, activate the offer"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "SalesPitchAgent")

        print(f"\n--- Turn 5: User decides on secondary offer ---")
        await asyncio.sleep(INTER_TURN_SLEEP)
        agent_message, _, _ = await run_turn(
            self.runner, self.user_id, session_id,
            make_resume_message(interrupt_id, "Yes please, include that too"), invocation_id
        )
        state = await self.get_session_state(session_id)
        self.assertEqual(state.get("current_agent"), "PostCallAgent")
        self.assertTrue(state.get("offer_accepted"))
        print(f"\n[PASS] Context break handled. Final agent: {state.get('current_agent')}")


if __name__ == "__main__":
    unittest.main()
