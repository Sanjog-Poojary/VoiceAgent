import asyncio
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
    is_appointment_accept: bool = False
    is_appointment_decline: bool = False
    is_injection_attempt: bool = False
    is_silent_turn: bool = False
    ambiguity_reason: str = ""
    confidence_score: float = 0.95

from google.adk.agents import Context
class DummyState:
    def __init__(self, data):
        self.data = data
    def get(self, k, default=None):
        return self.data.get(k, default)
    def to_dict(self):
        return self.data.copy()
    def __getitem__(self, k):
        return self.data[k]
    def __setitem__(self, k, v):
        self.data[k] = v

class DummyContext:
    def __init__(self):
        self.state = DummyState({
            "current_agent": "ClarifyingAgent",
            "previous_agent": "SpendingHistoryAgent",
            "offer_pitched": False,
            "last_outcome": "success",
            "agent_memory": {},
            "verification_attempts": 0
        })
        self.route = None
        
from orchestrator import orchestrator_node, TurnClassification as RealTurnClassification

async def run():
    ctx = DummyContext()
    # Mock classify_turn to return our hardcoded classification
    import orchestrator
    async def mock_classify(*args, **kwargs):
        print("MOCK CLASSIFY CALLED")
        return RealTurnClassification.model_validate(TurnClassification().model_dump())
    orchestrator.classify_turn = mock_classify
    
    await orchestrator_node.func(ctx, "yes I would like to hear the offer")
    print("FINAL AGENT:", ctx.route)

asyncio.run(run())
