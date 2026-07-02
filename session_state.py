from typing import List, Literal, Dict
from pydantic import BaseModel, Field

class AgentMemory(BaseModel):
    """Pydantic model representing structured sub-agent memory keys."""
    welcomed: bool = Field(default=False, description="Greeting welcomer status.")
    verified: bool = Field(default=False, description="Verification success status.")
    pitch_category: Literal["Fashion", "Beauty", "Luxury Watches", ""] = Field(default="", description="The retail category pitched.")
    offer_pitched: bool = Field(default=False, description="Personalized offer pitch status.")
    whatsapp_sent: bool = Field(default=False, description="WhatsApp notification status.")
    clarification_count: int = Field(default=0, description="Ambiguity clarification tracker.")

class SessionState(BaseModel):
    """Pydantic schema representing the Shoppers Stop AI Voice Agent session state.
    
    This model defines the structured state tracking for the conversation graph.
    All state mutations at runtime are validated against this model by the Google ADK.
    """
    # Customer Details
    customer_id: str = Field(default="", description="The unique ID of the Shoppers Stop customer.")
    detected_language: str = Field(default="English", description="The customer's language, dynamically updated during verification.")
    
    # Conversation State & Metadata
    current_agent: str = Field(default="GreetingAgent", description="The name of the currently active sub-agent.")
    verification_attempts: int = Field(default=0, description="The number of identity verification attempts.")
    call_sentiment: str = Field(default="Neutral", description="Current overall sentiment of the call (Positive, Neutral, Agitated).")
    
    # Offer Tracking Flags
    offer_pitched: bool = Field(default=False, description="Whether the personalized retail offer has been pitched to the user.")
    offer_accepted: bool = Field(default=False, description="Whether the customer verbally accepted the offer.")
    escalation_triggered: bool = Field(default=False, description="Whether highly dissatisfied state triggered CRM/WhatsApp escalation.")
    
    # Transcript
    raw_audio_transcription: List[str] = Field(
        default_factory=list, 
        description="Verbatim chronological transcript lines of the call session."
    )
    
    # Edge-case tracking
    silent_turns: int = Field(default=0, description="Consecutive turns where user produced no meaningful input (silence, ambient noise).")
    injection_attempts: int = Field(default=0, description="The number of prompt injection / adversarial override attempts.")
    escalation_reason: str = Field(default="agitated", description="The reason for escalation: 'agitated' or 'malicious'.")
    previous_agent: str = Field(default="", description="The previous agent active before an injection warning deflection occurred.")
    clarification_attempts: int = Field(default=0, description="The number of clarification attempts during ambiguous turns.")

    # Coordinator / Persistent Memory Layer
    current_goal: str = Field(default="", description="The current conversational goal or target agent's objective.")
    goal_history: List[str] = Field(default_factory=list, description="A historical log of conversational goals (max 5).")
    last_agent: str = Field(default="", description="The name of the last active sub-agent.")
    last_outcome: Literal["success", "failed", "accepted", "declined", "tangent", "silence", "pending", ""] = Field(default="", description="The outcome of the last agent's turn.")
    agent_memory: AgentMemory = Field(default_factory=AgentMemory, description="Schema-enforced persistent agent memory.")

    # Internal Critic / Decision Revision Layer (#4 + #5)
    revision_count: int = Field(default=0, description="Consecutive-turn critic revision counter. Increments when critic revises a route; resets to 0 only when critique is acceptable. Caps at 1 to prevent persistent re-revision on unresolved conversations.")
    revision_reason: Literal["", "route_context_mismatch", "outcome_contradicts_utterance", "unstated_precondition"] = Field(default="", description="The failure_reason from the last Critique that triggered a revision, or empty string if no revision occurred.")

