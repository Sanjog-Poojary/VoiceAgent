from typing import List, Literal, Dict
from pydantic import BaseModel, Field

class AgentMemory(BaseModel):
    """Pydantic model representing structured sub-agent memory keys."""
    welcomed: bool = Field(default=False, description="Greeting welcomer status.")
    verified: bool = Field(default=False, description="Verification success status.")
    pitch_category: Literal["Fashion", "Beauty", "Luxury Watches", ""] = Field(default="", description="The retail category pitched.")
    event_introduced: bool = Field(default=False, description="Whether the event intro (birthday/credit expiry) has been played.")
    offer_pitched: bool = Field(default=False, description="Personalized offer pitch status.")
    whatsapp_sent: bool = Field(default=False, description="WhatsApp notification status.")
    clarification_count: int = Field(default=0, description="Ambiguity clarification tracker.")
    has_secondary_offer: bool = Field(default=False, description="True if a secondary brand offer exists.")
    secondary_offer_pitched: bool = Field(default=False, description="True if the secondary offer was pitched.")
    primary_offer_accepted: bool = Field(default=False, description="True if the primary offer was accepted.")

class BoundedPlan(BaseModel):
    """Schema for individual sub-agent bounded multi-step plans."""
    current_objective: str = Field(..., description="The end goal (e.g., 'Secure Coupon Activation').")
    remaining_steps: List[str] = Field(default_factory=list, description="Steps left to accomplish the objective.")
    active_step: str = Field(default="", description="The step currently being executed.")
    step_history: List[str] = Field(default_factory=list, description="Steps already completed.")
    plan_status: str = Field(default="In Progress", description="e.g., 'In Progress', 'Completed', 'Abandoned'.")
    revision_count: int = Field(default=0, description="Tracks how many times the plan was modified due to tangents.")
    max_revisions: int = Field(default=3, description="Maximum number of allowable revisions before abandoning the plan.")
    is_resuming: bool = Field(default=False, description="Whether the plan is resuming from a tangent context break.")

class SessionState(BaseModel):
    """Pydantic schema representing the Shoppers Stop AI Voice Agent session state.
    
    This model defines the structured state tracking for the conversation graph.
    All state mutations at runtime are validated against this model by the Google ADK.
    """
    # Customer Details
    customer_id: str = Field(default="", description="The unique ID of the Shoppers Stop customer.")
    detected_language: str = Field(default="English", description="The customer's language, dynamically updated during verification.")
    
    # Conversation State & Metadata
    current_agent: str = Field(default="IdentityAgent", description="The name of the currently active sub-agent.")
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

    # Personal Shopper / Post-Call Action Layer
    personal_shopper_offered: bool = Field(default=False, description="True if the user was offered the personal shopper service.")
    personal_shopper_accepted: bool = Field(default=False, description="True if the user accepted the personal shopper service.")
    preferred_appointment_slot: str = Field(default="", description="Customer's stated preferred day/time for the appointment, raw text.")
    user_declined_offer: bool = Field(default=False, description="Persistent flag indicating the user explicitly declined the main offer.")
    last_knowledge_query: str = Field(default="", description="The specific RAG query asked by the user.")

    # Coordinator / Persistent Memory Layer
    current_goal: str = Field(default="", description="The current conversational goal or target agent's objective.")
    goal_history: List[str] = Field(default_factory=list, description="A historical log of conversational goals (max 5).")
    last_agent: str = Field(default="", description="The name of the last active sub-agent.")
    last_outcome: Literal["success", "failed", "accepted", "declined", "tangent", "silence", "pending", "interest", "knowledge_q", "secondary_pitch", "slot_captured", ""] = Field(default="", description="The outcome of the last agent's turn.")
    agent_memory: AgentMemory = Field(default_factory=AgentMemory, description="Schema-enforced persistent agent memory.")
    bounded_plans: Dict[str, BoundedPlan] = Field(default_factory=dict, description="Active multi-step plans mapped by agent name.")

    # Internal Critic / Decision Revision Layer
    revision_count: int = Field(default=0, description="Consecutive-turn critic revision counter. Increments when critic revises a route; resets to 0 only when critique is acceptable. Caps at 1 to prevent persistent re-revision on unresolved conversations.")
    revision_reason: Literal[
        "", 
        "route_context_mismatch", 
        "outcome_contradicts_utterance", 
        "unstated_precondition",
        "low_confidence",
        "goal_misalignment",
        "premature_termination",
        "ambiguous_intent"
    ] = Field(default="", description="The failure_reason from the last Critique that triggered a revision, or empty string if no revision occurred.")
    
    reflection_enabled: bool = Field(default=True, description="Whether the critic pass is active. Can be toggled for debugging or A/B testing.")


