import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_server")

app = FastAPI(
    title="Shoppers Stop Mock API Server",
    description="Development API server for customer engagement, events, offers, and notifications.",
    version="1.0.0"
)

# Mock Data
CUSTOMERS = {
    "1": {"id": "1", "name": "Sanjog", "phone": "+1234567890", "base_language": "English", "preferred_category": "Fashion"},
    "2": {"id": "2", "name": "Aarav", "phone": "+919876543210", "base_language": "Hindi", "preferred_category": "Beauty"},
    "3": {"id": "3", "name": "Ananya", "phone": "+918888888888", "base_language": "English", "preferred_category": "Luxury Watches"}
}

EVENTS = {
    "1": {"id": "ev_1", "customer_id": "1", "event_type": "Birthday", "event_date": "2026-06-29"},
    "2": {"id": "ev_2", "customer_id": "2", "event_type": "Credit Expiry", "event_date": "2026-07-02"},
    "3": {"id": "ev_3", "customer_id": "3", "event_type": "Birthday", "event_date": "2026-06-30"}
}

OFFERS = {
    "1": {
        "id": "off_1",
        "customer_id": "1",
        "category": "Fashion",
        "discount_percentage": 20,
        "coupon_code": "BIRTHDAY20",
        "recommendations": ["Formal Shirt", "Jeans", "Sneakers"]
    },
    "2": {
        "id": "off_2",
        "customer_id": "2",
        "category": "Beauty",
        "discount_percentage": 15,
        "coupon_code": "CREDIT15",
        "recommendations": ["Calvin Klein One", "Bvlgari Man"]
    },
    "3": {
        "id": "off_3",
        "customer_id": "3",
        "category": "Luxury Watches",
        "discount_percentage": 25,
        "coupon_code": "LUX25",
        "recommendations": ["Fossil Watch", "Seiko Automatic"]
    }
}

# Schemas
class Customer(BaseModel):
    id: str
    name: str
    phone: str
    base_language: str
    preferred_category: str

class Event(BaseModel):
    id: str
    customer_id: str
    event_type: str
    event_date: str

class Offer(BaseModel):
    id: str
    customer_id: str
    category: str
    discount_percentage: int
    coupon_code: str
    recommendations: List[str]

class WhatsAppNotification(BaseModel):
    customer_id: str
    phone: str
    message: str

class CRMTicket(BaseModel):
    customer_id: str
    issue_description: str
    priority: str = "medium"

class AppointmentRequest(BaseModel):
    customer_id: str
    preferred_slot: str

# Endpoints
@app.get("/api/users/{customer_id}", response_model=Customer)
def get_user(customer_id: str):
    logger.info(f"Fetching user details for customer_id: {customer_id}")
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@app.get("/api/events/{customer_id}", response_model=Event)
def get_event(customer_id: str):
    logger.info(f"Fetching event triggers for customer_id: {customer_id}")
    event = EVENTS.get(customer_id)
    if not event:
        raise HTTPException(status_code=404, detail="No active events for customer")
    return event

@app.get("/api/offers", response_model=List[Offer])
def get_all_offers():
    logger.info("Fetching all active store offers")
    return list(OFFERS.values())

@app.get("/api/offers/{customer_id}", response_model=Offer)
def get_offer(customer_id: str):
    logger.info(f"Fetching personalized offer for customer_id: {customer_id}")
    offer = OFFERS.get(customer_id)
    if not offer:
        raise HTTPException(status_code=404, detail="No personalized offers for customer")
    return offer

@app.post("/api/notify/whatsapp")
def send_whatsapp(payload: WhatsAppNotification):
    logger.info(f"Simulating WhatsApp message to customer {payload.customer_id} ({payload.phone}): {payload.message}")
    return {"status": "success", "message": f"WhatsApp message successfully sent to {payload.phone}"}

@app.post("/api/tickets/crm")
def create_crm_ticket(payload: CRMTicket):
    logger.info(f"Generating CRM ticket for customer {payload.customer_id}: [{payload.priority.upper()}] {payload.issue_description}")
    return {"status": "success", "ticket_id": f"ticket_{payload.customer_id}_999", "message": "CRM ticket created successfully"}

@app.post("/api/appointments/personal-shopper")
def create_appointment(payload: AppointmentRequest):
    logger.info(f"Creating personal shopper appointment for customer {payload.customer_id} at {payload.preferred_slot}")
    return {"status": "success", "appointment_id": f"apt_{payload.customer_id}_123"}


# ==========================================
# ADK AGENT WORKFLOW INTEGRATION ENDPOINTS
# ==========================================
import os
import uuid
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.apps import App, ResumabilityConfig
try:
    from orchestrator import VoiceAgentWorkflow
except ModuleNotFoundError:
    from VoiceAgent.orchestrator import VoiceAgentWorkflow
from fastapi.responses import HTMLResponse

# Initialize workflow runners
session_service = InMemorySessionService()
workflow = VoiceAgentWorkflow(name="voice_agent_workflow")
adk_app = App(
    name="VoiceAgent",
    root_agent=workflow,
    resumability_config=ResumabilityConfig(is_resumable=True)
)
runner = Runner(
    app=adk_app,
    session_service=session_service,
    auto_create_session=True
)

class ChatStartRequest(BaseModel):
    customer_id: str

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    interrupt_id: Optional[str] = None
    invocation_id: Optional[str] = None

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

@app.post("/api/chat/start")
async def chat_start(payload: ChatStartRequest):
    session_id = f"session_{uuid.uuid4().hex}"
    
    # Initialize state delta
    initial_state = {
        "customer_id": payload.customer_id,
        "detected_language": "English",
        "current_agent": "GreetingAgent",
        "verification_attempts": 0,
        "call_sentiment": "Neutral",
        "offer_pitched": False,
        "offer_accepted": False,
        "escalation_triggered": False,
        "raw_audio_transcription": []
    }
    
    agent_message = ""
    invocation_id = None
    interrupt_id = None
    
    # Run the first turn
    async for event in runner.run_async(
        user_id="web_tester",
        session_id=session_id,
        new_message=make_user_message("[Call Connected]"),
        state_delta=initial_state
    ):
        if event.invocation_id:
            invocation_id = event.invocation_id
        iid = get_interrupt_id(event)
        if iid:
            interrupt_id = iid
        msg = get_agent_message_text(event)
        if msg:
            agent_message = msg
            
    # Get current session state
    session = await session_service.get_session(
        app_name="VoiceAgent",
        user_id="web_tester",
        session_id=session_id
    )
    state = session.state if session else {}
    
    return {
        "session_id": session_id,
        "message": agent_message,
        "interrupt_id": interrupt_id,
        "invocation_id": invocation_id,
        "state": state
    }

@app.post("/api/chat/message")
async def chat_message(payload: ChatMessageRequest):
    agent_message = ""
    next_interrupt_id = None
    
    # Form input message depending on whether it's resuming an interrupt
    if payload.interrupt_id:
        new_msg = make_resume_message(payload.interrupt_id, payload.message)
    else:
        new_msg = make_user_message(payload.message)
        
    async for event in runner.run_async(
        user_id="web_tester",
        session_id=payload.session_id,
        new_message=new_msg,
        invocation_id=payload.invocation_id
    ):
        iid = get_interrupt_id(event)
        if iid:
            next_interrupt_id = iid
        msg = get_agent_message_text(event)
        if msg:
            agent_message = msg
            
    # Get current session state
    session = await session_service.get_session(
        app_name="VoiceAgent",
        user_id="web_tester",
        session_id=payload.session_id
    )
    state = session.state if session else {}
    
    return {
        "message": agent_message,
        "interrupt_id": next_interrupt_id,
        "state": state
    }

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h3>index.html not found</h3>", status_code=404)
