from typing import List
from pydantic import BaseModel, Field

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

