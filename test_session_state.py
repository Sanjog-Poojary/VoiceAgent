
import unittest
from pydantic import ValidationError
from session_state import SessionState
from google.adk.sessions.state import State, StateSchemaError

class TestSessionState(unittest.TestCase):
    
    def test_session_state_pydantic_defaults(self):
        """Verify that SessionState initializes with correct default values."""
        state = SessionState()
        self.assertEqual(state.customer_id, "")
        self.assertEqual(state.detected_language, "English")
        self.assertEqual(state.current_agent, "IdentityAgent")
        self.assertEqual(state.verification_attempts, 0)
        self.assertEqual(state.call_sentiment, "Neutral")
        self.assertFalse(state.offer_pitched)
        self.assertFalse(state.offer_accepted)
        self.assertFalse(state.escalation_triggered)
        self.assertEqual(state.raw_audio_transcription, [])

    def test_session_state_pydantic_validation(self):
        """Verify standard Pydantic validation on fields."""
        # Correct types should pass
        state = SessionState(customer_id="SS_123", verification_attempts=2)
        self.assertEqual(state.customer_id, "SS_123")
        self.assertEqual(state.verification_attempts, 2)

        # Incorrect types (e.g. invalid type for boolean) should fail validation.
        with self.assertRaises(ValidationError):
            SessionState(offer_pitched="not-a-bool")

    def test_adk_state_integration_valid(self):
        """Verify integration with google.adk.sessions.state.State for valid mutations."""
        initial_values = {}
        delta = {}
        
        # Initialize State with our schema
        adk_state = State(value=initial_values, delta=delta, schema=SessionState)
        
        # Mutating valid keys with correct types should work
        adk_state["customer_id"] = "SS_999"
        adk_state["detected_language"] = "Hindi"
        adk_state["verification_attempts"] = 1
        adk_state["offer_pitched"] = True
        
        self.assertEqual(adk_state["customer_id"], "SS_999")
        self.assertEqual(adk_state["detected_language"], "Hindi")
        self.assertEqual(adk_state["verification_attempts"], 1)
        self.assertTrue(adk_state["offer_pitched"])
        self.assertEqual(adk_state.to_dict()["customer_id"], "SS_999")

    def test_adk_state_integration_invalid(self):
        """Verify integration with google.adk.sessions.state.State for invalid mutations."""
        initial_values = {}
        delta = {}
        
        adk_state = State(value=initial_values, delta=delta, schema=SessionState)
        
        # Mutation of an invalid key (not in schema) should raise StateSchemaError
        with self.assertRaises(StateSchemaError) as context:
            adk_state["non_existent_key"] = "some_value"
        self.assertIn("not declared in state schema", str(context.exception))
        
        # Mutation of a valid key with incorrect type should raise StateSchemaError
        with self.assertRaises(StateSchemaError) as context:
            adk_state["verification_attempts"] = "three"
        self.assertIn("does not match type", str(context.exception))

if __name__ == "__main__":
    unittest.main()
