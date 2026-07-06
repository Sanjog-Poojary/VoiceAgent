import json
from pydantic import BaseModel
class TurnClassification(BaseModel):
    detected_language: str = "English"
    call_sentiment: str = "Positive"
    is_valid_answer: bool = False
    is_acceptance: bool = True
    is_decline: bool = False
    is_third_party: bool = False
    is_competitor_mention: bool = False
    is_loyalty_question: bool = False
    is_injection_attempt: bool = False
    is_silent_turn: bool = False
    ambiguity_reason: str = ""
    confidence_score: float = 0.95
    is_appointment_accept: bool = False
    is_appointment_decline: bool = False

classification = TurnClassification()
state = {
    "current_agent": "ClarifyingAgent",
    "previous_agent": "SpendingHistoryAgent",
    "offer_pitched": False,
    "last_outcome": "success"
}
user_input_str = "yes I would like to hear the offer"

from orchestrator import apply_deterministic_rules
next_agent, updates = apply_deterministic_rules(classification, state, user_input_str)
print("FINAL FROM RULES:", next_agent)
