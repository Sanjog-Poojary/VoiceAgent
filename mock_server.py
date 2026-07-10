import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_server")

# Load PDF knowledge files
PDF_FILES = [
    "delivery_policy.pdf",
    "first_citizen_policy.pdf",
    "privacy_policy.pdf",
    "returns_exchange_policy.pdf",
    "shoppersstop_faq.pdf",
    "terms_and_conditions.pdf"
]

pdf_text_parts = []
for filename in PDF_FILES:
    if os.path.exists(filename):
        try:
            reader = PdfReader(filename)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            pdf_text_parts.append(f"--- DOCUMENT: {filename} ---\n{text}")
            logger.info(f"Loaded knowledge document {filename} ({len(text)} chars)")
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")

PDF_KNOWLEDGE_TEXT = "\n\n".join(pdf_text_parts)

_GENAI_CLIENT = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
)

app = FastAPI(
    title="Shoppers Stop Mock API Server",
    description="Development API server for customer engagement, events, offers, and notifications.",
    version="1.0.0"
)

# Mock Data
CUSTOMERS = {
    "1": {"id": "1", "name": "Sanjog", "phone": "+1234567890", "base_language": "English", "preferred_category": "Fashion", "secondary_brand": "Puma"},
    "2": {"id": "2", "name": "Aarav", "phone": "+919876543210", "base_language": "Hindi", "preferred_category": "Beauty", "secondary_brand": "MAC"},
    "3": {"id": "3", "name": "Ananya", "phone": "+918888888888", "base_language": "English", "preferred_category": "Luxury Watches", "secondary_brand": "Bobbi Brown"}
}

EVENTS = {
    "1": {"id": "ev_1", "customer_id": "1", "event_type": "Birthday", "event_date": "2026-06-29"},
    "2": {"id": "ev_2", "customer_id": "2", "event_type": "Credit Expiry", "event_date": "2026-07-02"},
    "3": {"id": "ev_3", "customer_id": "3", "event_type": "Birthday", "event_date": "2026-06-30"}
}

OFFERS = {
    "1": {
        "offer_id": "off_1",
        "offer_name": "BIRTHDAY20",
        "offer_brand": "Stop",
        "offer_category": "Fashion",
        "valid_from": "2026-06-29",
        "valid_to": "2026-07-29",
        "offer_description": "Get 20% off on Stop everyday casuals and formal wear."
    },
    "2": {
        "offer_id": "off_2",
        "offer_name": "CREDIT15",
        "offer_brand": "Arcelia",
        "offer_category": "Beauty",
        "valid_from": "2026-07-02",
        "valid_to": "2026-08-02",
        "offer_description": "Get 15% off on Arcelia fragrances and makeup products."
    },
    "3": {
        "offer_id": "off_3",
        "offer_name": "LUX25",
        "offer_brand": "Michael Kors",
        "offer_category": "Luxury Watches",
        "valid_from": "2026-06-30",
        "valid_to": "2026-07-30",
        "offer_description": "Get 25% off on Michael Kors luxury watches and accessories."
    },
    "4": {
        "offer_id": "off_4",
        "offer_name": "PUMA15",
        "offer_brand": "Puma",
        "offer_category": "Activewear",
        "valid_from": "2026-07-01",
        "valid_to": "2026-08-01",
        "offer_description": "Get 15% off all Puma running shoes and apparel."
    }
}

# Schemas
class Customer(BaseModel):
    id: str
    name: str
    phone: str
    base_language: str
    preferred_category: str
    secondary_brand: str = ""

class Event(BaseModel):
    id: str
    customer_id: str
    event_type: str
    event_date: str

class OfferItem(BaseModel):
    offer_id: str
    offer_name: str
    offer_brand: str
    offer_category: str
    valid_from: str
    valid_to: str
    offer_description: str

class OffersResponse(BaseModel):
    no_of_offers: int
    offers: List[OfferItem]

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

@app.get("/api/offers", response_model=OffersResponse)
def get_all_offers():
    logger.info("Fetching all active store offers")
    return {"no_of_offers": len(OFFERS), "offers": list(OFFERS.values())}

@app.get("/api/offers/{customer_id}", response_model=OfferItem)
def get_offer(customer_id: str):
    logger.info(f"Fetching personalized offer for customer_id: {customer_id}")
    offer = OFFERS.get(customer_id)
    if not offer:
        raise HTTPException(status_code=404, detail="No personalized offers for customer")
    return offer

WHATSAPP_LOGS = []
CRM_LOGS = []
APPOINTMENT_LOGS = []

@app.post("/api/notify/whatsapp")
def send_whatsapp(payload: WhatsAppNotification):
    logger.info(f"Simulating WhatsApp message to customer {payload.customer_id} ({payload.phone}): {payload.message}")
    entry = {"customer_id": payload.customer_id, "phone": payload.phone, "message": payload.message, "status": "success"}
    WHATSAPP_LOGS.append(entry)
    return {"status": "success", "message": f"WhatsApp message successfully sent to {payload.phone}"}

@app.get("/api/notify/whatsapp/logs")
def get_whatsapp_logs():
    return WHATSAPP_LOGS

@app.post("/api/tickets/crm")
def create_crm_ticket(payload: CRMTicket):
    logger.info(f"Generating CRM ticket for customer {payload.customer_id}: [{payload.priority.upper()}] {payload.issue_description}")
    entry = {
        "customer_id": payload.customer_id,
        "issue_description": payload.issue_description,
        "priority": payload.priority,
        "ticket_id": f"ticket_{payload.customer_id}_999",
        "status": "success"
    }
    CRM_LOGS.append(entry)
    return {"status": "success", "ticket_id": f"ticket_{payload.customer_id}_999", "message": "CRM ticket created successfully"}

@app.get("/api/tickets/crm/logs")
def get_crm_logs():
    return CRM_LOGS

@app.post("/api/appointments/personal-shopper")
def create_appointment(payload: AppointmentRequest):
    logger.info(f"Creating personal shopper appointment for customer {payload.customer_id} at {payload.preferred_slot}")
    entry = {
        "customer_id": payload.customer_id,
        "preferred_slot": payload.preferred_slot,
        "status": "success",
        "appointment_id": f"apt_{payload.customer_id}_123"
    }
    APPOINTMENT_LOGS.append(entry)
    return {"status": "success", "appointment_id": f"apt_{payload.customer_id}_123"}

@app.get("/api/appointments/personal-shopper/logs")
def get_appointment_logs():
    return APPOINTMENT_LOGS


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

KNOWLEDGE_BASE = {
    "mac": "Standard promotions exclude premium beauty brands like MAC and Jo Malone.",
    "exclusion": "Our standard offers apply to most categories, but premium luxury brands and cosmetics are generally excluded.",
    "tailor": "Shoppers Stop offers free basic alterations for our First Citizen loyalty members at all major outlets.",
    "return": "You can exchange apparel within 14 days at any Shoppers Stop store, provided the tags are intact.",
    "parking": "Most of our mall locations, including Inorbit Malad, offer valet parking and remain open until 9:30 PM.",
    "loyalty": "Points earned today will upgrade your tier immediately, and they do not expire for 12 months from the date of purchase.",
    "tom ford": "Yes, we carry Tom Ford fragrances in our SSBeauty premium sections, though they are excluded from standard discount codes.",
    "football": "Our activewear section carries professional performance gear, including football studs from both Puma and Adidas.",
    "online": "Yes, you can apply this promo code on our mobile app and select the 'Buy Online, Pick Up In Store' option at checkout."
}

@app.get("/api/knowledge")
def query_knowledge(q: str = ""):
    logger.info(f"Querying knowledge base for: {q}")
    query = q.lower().strip()
    
    # 1. Local keyword matches (fast & guaranteed for E2E scenarios)
    # Check specific returns first to avoid collision with general "return" key
    if "return" in query or "exchange" in query:
        if any(w in query for w in ("perfume", "cosmetic", "beauty", "fragrance", "deodorant")):
            ans = "Shoppers Stop does not accept returns or exchanges on perfumes, cosmetics, or innerwear due to hygiene reasons."
            logger.info(f"Found specific perfume/cosmetic return match: {ans}")
            return {"status": "success", "answer": ans}

    for key, answer in KNOWLEDGE_BASE.items():
        if key in query:
            logger.info(f"Found local match for '{key}': {answer}")
            return {"status": "success", "answer": answer}
            
    # 2. Dynamic PDF RAG query via Gemini fallback
    if PDF_KNOWLEDGE_TEXT:
        logger.info("Local key match not found. Querying Gemini with PDF policies context...")
        prompt = (
            "You are the expert Shoppers Stop virtual assistant answering customer questions on outbound calls.\n"
            "Use the provided Shoppers Stop official policies text to answer the customer's question.\n"
            "Keep your answer brief, friendly, conversational, and exactly 1-2 sentences. Do not mention section numbers or document names.\n"
            "If the answer cannot be found in the provided text, reply with: "
            "'I don't have the exact details on that right now, but our store staff will be happy to help!'\n\n"
            f"Shoppers Stop Policies:\n{PDF_KNOWLEDGE_TEXT}\n\n"
            f"Customer Question: {q}\n"
            "Answer:"
        )
        try:
            response = _GENAI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            ans = response.text.strip()
            logger.info(f"Gemini RAG generated response: {ans}")
            return {"status": "success", "answer": ans}
        except Exception as e:
            logger.error(f"Error querying Gemini RAG fallback: {e}")
            
    return {"status": "success", "answer": "I don't have the exact details on that right now, but our store staff will be happy to help!"}

