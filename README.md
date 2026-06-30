# Shoppers Stop AI Voice Agent

An outbound AI voice agent for Shoppers Stop built on **Google Agent Development Kit (ADK)** + **Groq LLM** + **FastAPI**. The agent places personalised retention calls to loyalty customers, verifies their identity, delivers contextual event notifications (birthdays, credit expiry), pitches tailored retail offers, and handles edge-cases with production-grade guardrails.

---

## Architecture

```
Browser / ADK Web UI
        │
        ▼
  FastAPI Mock Server  ◄──────────────────────────────────────────┐
  (mock_server.py)                                                │
        │                                                         │
        ▼                                                         │
  ADK Workflow Graph (orchestrator.py)                            │
  ┌─────────────────────────────────────────────────────────┐     │
  │                                                         │     │
  │  [Call Connected]                                       │     │
  │       │                                                 │     │
  │  GreetingAgent ──── ask identity ───────────────────────┤     │
  │       │                                                 │     │
  │  VerificationAgent ─── confirm name ────────────────────┤     │
  │       │                                                 │     │
  │  EventAgent ─── birthday / credit-expiry message ───────┤     │
  │       │                                                 │     │
  │  SpendingHistoryAgent ─── category preference ──────────┤     │
  │       │                                                 │     │
  │  OfferAgent ─── personalised coupon pitch ──────────────┤     │
  │       │                                                 │     │
  │  PostCallAgent ─── activate & WhatsApp confirm ─────────┤     │
  │       │                                                 │     │
  │  [Terminate]                                            │     │
  │                                                         │     │
  │  ─── EscalationAgent (any point)                        │     │
  │  ─── ApologyAgent   (any point)                         │     │
  └─────────────────────────────────────────────────────────┘     │
        │                                                         │
        ▼                                                         │
  Orchestrator LLM (Groq llama-3.1-8b-instant)                   │
  + Deterministic Python Override Layer                           │
        │                                                         │
        └─────────────── REST API calls ──────────────────────────┘
```

Each agent node yields a `RequestInput` (turn-taking) and hands control back to the **orchestrator** which makes an LLM routing decision, then applies a deterministic Python override layer for safety-critical routing.

---

## Features Implemented

### Core Agent Flow
| Agent | Responsibility |
|---|---|
| `GreetingAgent` | Opens call, greets customer by name |
| `VerificationAgent` | Confirms caller identity, tracks `verification_attempts` |
| `EventAgent` | Delivers contextual event (birthday wish, credit expiry alert) |
| `SpendingHistoryAgent` | Fetches preferred shopping category via `/api/users/{id}` |
| `OfferAgent` | Pitches personalised coupon from `/api/offers` filtered by category |
| `PostCallAgent` | Activates offer, sends WhatsApp confirmation via `/api/webhooks/whatsapp` |
| `ApologyAgent` | Politely closes call on rejection, mismatch, or out-of-domain request |
| `EscalationAgent` | Creates CRM ticket via `/api/crm/create-ticket`, queues supervisor transfer |

### Offer Personalisation
- `/api/offers` endpoint serves a catalogue of offers per category (Fashion, Beauty, Luxury Watches)
- `SpendingHistoryAgent` fetches the customer's `preferred_category` from `/api/users/{id}`
- `OfferAgent` filters the catalogue to pitch only the offer relevant to that customer's taste

### Session State Schema (`session_state.py`)
All state is typed, validated at runtime by ADK's `StateSchemaError` engine:

| Field | Type | Purpose |
|---|---|---|
| `customer_id` | str | Links session to mock CRM |
| `detected_language` | str | `English` / `Hindi` — updated dynamically |
| `current_agent` | str | Active sub-agent name |
| `verification_attempts` | int | Identity check retry counter |
| `call_sentiment` | str | `Neutral` / `Positive` / `Agitated` |
| `offer_pitched` | bool | Whether offer was presented |
| `offer_accepted` | bool | Whether customer verbally accepted |
| `escalation_triggered` | bool | Whether CRM escalation was filed |
| `raw_audio_transcription` | list[str] | Chronological call transcript |
| `silent_turns` | int | Consecutive non-verbal turns |
| `injection_attempts` | int | Adversarial override attempt counter |
| `escalation_reason` | str | `agitated` or `malicious` |
| `previous_agent` | str | Recovery target after injection warning |

---

## Orchestrator Guardrails

The orchestrator applies **8 deterministic safety rules** on top of every LLM routing decision:

| # | Guardrail | Trigger | Action |
|---|---|---|---|
| 1 | Language Switch | Hindi keywords detected | Force `detected_language = Hindi` |
| 2 | Prompt Injection — 1st attempt | Injection markers in input | Warn via `ApologyAgent`, save `previous_agent`, do **not** escalate |
| 2b | Prompt Injection — 2nd attempt | Injection markers again | `escalation_reason = malicious`, route `EscalationAgent` |
| 3 | Agitated Keywords | "gussa", "supervisor", "angry" etc. | Immediate `EscalationAgent` |
| 3.5 | Sentiment Enforcer | LLM returns `Agitated` or `escalation_triggered` | Override routing to `EscalationAgent` |
| 4 | Silent / Ambient User | `...`, "sound of wind" etc. | Increment `silent_turns`; after 2 → `ApologyAgent` |
| 5a | Third-Party Gatekeeper | "I am her husband" etc. | Route to `ApologyAgent` without pitching offer |
| 5b | Verification Loop | `verification_attempts >= 2` with no confirmation | Exit to `ApologyAgent` |
| 5c | Competitor Baiter | Competitor brand names | `ApologyAgent`, no competitor data leaked |
| 5d | Context Breaker | Loyalty question mid-offer | Detour to `SpendingHistoryAgent`, return to close offer |

---

## Mock API Endpoints (`mock_server.py`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/users/{customer_id}` | Customer profile + preferred category |
| `GET` | `/api/events/{customer_id}` | Event triggers (birthday, credit expiry) |
| `GET` | `/api/offers` | Full offer catalogue |
| `POST` | `/api/chat/start` | Start ADK session |
| `POST` | `/api/chat/message` | Send turn message |
| `POST` | `/api/webhooks/whatsapp` | WhatsApp confirmation webhook |
| `POST` | `/api/crm/create-ticket` | CRM escalation ticket creation |

---

## Edge Case Test Suite (`test_orchestrator.py`)

11 automated scenarios run against a live LLM:

| Scenario | Description | Key Assertion |
|---|---|---|
| A | Happy Path (English) | Full flow from greeting to `Terminate` |
| B | Hindi + Escalation | Language switch + supervisor escalation |
| C | Suspicious Gatekeeper | Offer NOT pitched to unverified 3rd party |
| D | Ambiguous Identity Loop | Loop exits after 2 ambiguous replies |
| E | Mid-Call Language Switch | `detected_language` updates to Hindi |
| F | Internet Slang | No premature OfferAgent routing |
| G | Competitor Baiter | No competitor names in response |
| H | Prompt Injector | 1st attempt: warning deflection; 2nd: `malicious` escalation |
| I | Sarcastic Spender | Sarcasm classified as `Agitated`, not `Positive` |
| J | Silent / Ambient User | No offer pitched to silent caller |
| K | Context Breaker | Offer accepted after loyalty tangent |

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Groq API key

### Install
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Environment
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the Mock Server
```bash
uvicorn mock_server:app --host 127.0.0.1 --port 8001 --reload
```

### Run the ADK Web UI
```bash
adk web --port 8002
```
Open `http://localhost:8002` and select the `VoiceAgent` app.

### Run Tests (individual scenario)
```bash
python -m unittest test_orchestrator.TestVoiceAgentOrchestrator.test_scenario_a_happy_path
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent Framework | Google Agent Development Kit (ADK) |
| LLM | Groq `llama-3.1-8b-instant` |
| API Server | FastAPI + Uvicorn |
| Session State | ADK `InMemorySessionService` + Pydantic |
| Frontend UI | Vanilla HTML/CSS/JS (`index.html`) |
| Testing | Python `unittest.IsolatedAsyncioTestCase` |

---

## Project Structure

```
VoiceAgent/
├── orchestrator.py          # Agent workflow graph, all nodes, orchestrator LLM + guardrails
├── mock_server.py           # FastAPI mock CRM/event/offer/chat server
├── session_state.py         # Pydantic SessionState schema
├── agent.py                 # ADK app entrypoint
├── index.html               # Frontend voice agent UI
├── test_orchestrator.py     # 11-scenario automated LLM test suite
├── requirements.txt         # Python dependencies
├── PRD.md                   # Product Requirements Document
└── .env                     # API keys (not committed)
```
