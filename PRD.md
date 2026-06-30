# Master PRD: Shoppers Stop Personalized AI Voice Agent

## 1. System Overview

**Project:** Autonomous Outbound Voice Agent for Customer Engagement.
**Objective:** A personalized, voice-driven AI agent that initiates calls to existing Shoppers Stop customers on special occasions (birthdays, credit expiry) to pitch tailored retail offers based on their spending history.
**Core Framework:** Google ADK (Agent Development Kit).
**Architecture Pattern:** Multi-Agent Supervisor / Router Graph.

## 2. Technical Stack & Environment Strategy

To maximize development velocity and optimize for local testing, the LLM architecture is split into two distinct phases:

* **Development Phase (Logic & State Routing):** Groq API (via LiteLLM proxy if necessary). Used strictly for text-in/text-out orchestration, rapid prompting iterations, and testing ADK state transitions without cloud latency.
* **Production Phase (Audio Integration):** Google Cloud Vertex AI (Gemini Live API). Swapped in during the final build phase to handle native WebSockets and raw bidirectional audio via ADK's `LiveRequestQueue`.
* **Agent Framework:** Google ADK
* **Data Integration:** REST API (Internal Mock FastAPI server for development)
* **State & Memory:** ADK Artifacts (Session state tracking and structured JSON output)

## 3. Core Guardrails & System Constraints

* **Dynamic Multilingualism:** All calls initiate in English. The system must actively detect the user's spoken language during the verification phase. The Orchestrator must immediately update the session state to lock in the user's preferred language for all subsequent sub-agent responses.
* **Out-of-Bounds Handling:** The agent must operate strictly within Shoppers Stop's retail domains (fashion, beauty, home decor, etc.). If a user asks out-of-context questions, the agent must gracefully decline to answer and immediately pivot back to the relevant offer/event using a valid follow-up question.
* **Latency & Time Constraints:** The routing and REST API lookups must be optimized for low-latency conversational cadence.

## 4. Multi-Agent Roster & Routing Logic

The system consists of one (1) background routing agent and nine (9) specialized sub-agents. The Orchestrator does not speak; it analyzes user transcripts and delegates execution to the appropriate sub-agent.

1. **Orchestrator Agent (Supervisor):**
* *Role:* The central brain. Tracks session state (language, sentiment, intent), evaluates user transcripts, handles conversational state transitions, and generates logging/routing rationales.


2. **Greeting Agent:**
* *Role:* Initiates the call. (e.g., "Hello, am I speaking with [Customer Name]?")


3. **Verification Agent:**
* *Role:* Confirms identity and detects the user's language based on their initial response.


4. **Event Agent:**
* *Role:* Delivers the specific occasion context (e.g., Birthday wishes, loyalty point expiration alerts).


5. **Spending History Agent:**
* *Role:* Acknowledges the user's past affinity with the brand to build rapport (e.g., referencing favorite categories or tier status).


6. **Offer Agent:**
* *Role:* Queries the mock REST API and verbally pitches the dynamically personalized discount or promotion.


7. **Apology Agent (Friction State):**
* *Role:* Deployed if the Orchestrator detects minor dissatisfaction or disinterest. Apologizes gracefully and attempts to safely wrap up the call without burning the bridge.


8. **Escalation Agent (Critical State):**
* *Role:* Triggered by highly agitated or dissatisfied users. Aborts the sales pitch, terminates the call gracefully, and fires two tool calls:
* Send an email ticket to the CRM team.
* Trigger an automated apology WhatsApp message via API.




9. **Post-Call Agent (Success State):**
* *Role:* Triggered after a successful interaction. Fires a tool call to send a personalized WhatsApp greeting summarizing the offer.



## 5. API & Tool Integrations (REST)

Since no MCP server is provided, the agent will utilize standard REST APIs (mocked in development).

* **`GET /api/users/{customer_id}`:** Hydrates the system with Name, Phone, and Base Language.
* **`GET /api/events/{customer_id}`:** Fetches current triggers (Birthday, Credit Expiry).
* **`GET /api/offers/{customer_id}`:** Returns specific product recommendations and discount codes.
* **`POST /api/notify/whatsapp`:** Sends Post-Call or Escalation messages.
* **`POST /api/tickets/crm`:** Generates an email ticket for the CRM dashboard.

## 6. Memory & Database Logging (ADK Artifacts)

The session will utilize Google ADK's `Artifacts` to maintain state without polluting the context window. Upon call termination, the system must export a comprehensive log to the database containing:

1. **Raw Audio Transcription:** The full verbatim text log of the conversation.
2. **Inference JSON Payload:** Structured data extracted by the ADK Artifact, including:
* `customer_id`
* `detected_language`
* `call_sentiment` (Positive, Neutral, Agitated)
* `offer_pitched` (Boolean)
* `offer_accepted` (Boolean)
* `escalation_triggered` (Boolean)



## 7. AI Development Iteration Plan

*(Instructions for the AI Coding Assistant)*

* **Phase 1 (Groq API):** Scaffold the Mock FastAPI server for all required REST endpoints.
* **Phase 2 (Groq API):** Define the ADK `Artifact` schema for session state.
* **Phase 3 (Groq API):** Build the Orchestrator Agent logic and test state transitions using dummy text transcripts to ensure routing functions perfectly.
* **Phase 4 (Groq API):** Scaffold and bind tools to the core flow agents (Greeting, Verification, Event, Spending, Offer) testing purely via text.
* **Phase 5 (Groq API):** Scaffold and bind tools to the resolution agents (Apology, Escalation, Post-Call).
* **Phase 6 (Gemini API Swap):** Replace the Groq endpoint with the Gemini Live API. Implement the final bidirectional voice streaming via WebSockets and build the database JSON export formatting.