# Copilot Instructions — Shoppers Stop AI Voice Agent

## Project Overview
This is an outbound AI voice agent for Shoppers Stop built with Google Agent Development Kit (ADK), Groq LLM (llama-3.1-8b-instant), and FastAPI. It routes customer calls through a graph of sub-agents.

## Key Files
- `orchestrator.py` — The complete ADK workflow graph. All agent nodes (`GreetingAgent`, `VerificationAgent`, `EventAgent`, `SpendingHistoryAgent`, `OfferAgent`, `PostCallAgent`, `ApologyAgent`, `EscalationAgent`) are defined here, along with the `orchestrator_node` which makes LLM-based routing decisions overlaid with deterministic Python guardrails.
- `mock_server.py` — FastAPI server providing mock CRM, event, offer, and chat endpoints.
- `session_state.py` — Pydantic `SessionState` model. **Every new field added to ctx.state must also be declared here** or ADK raises `StateSchemaError`.
- `test_orchestrator.py` — 11 adversarial LLM integration test scenarios.
- `index.html` — Frontend voice agent UI served by the mock server.

## Architecture Rules
1. **State Schema is strict**: Always add new fields to `SessionState` in `session_state.py` AND to `init_state_defaults()` in `orchestrator.py`.
2. **Orchestrator routing is two-layer**: LLM decision first, then deterministic Python overrides. The Python layer always wins for safety-critical routes (escalation, injection, verification loops).
3. **No code execution in agent responses**: Agent nodes only yield `RequestInput(message=msg)` — never raw Python output.
4. **Injection defense is graduated**: 1st attempt → warn + deflect to `ApologyAgent`; 2nd attempt → `escalation_reason = "malicious"` + route to `EscalationAgent`.
5. **Sentiment Enforcer**: If `call_sentiment == "Agitated"` at any point, the enforcer overrides `next_agent = "EscalationAgent"` unless it's the first injection warning turn.

## Coding Conventions
- All agent nodes are decorated with `@node(name="AgentName")` and are `async def`.
- Use `ctx.state.get("field", default)` to read state safely.
- Use `ctx.state["field"] = value` to write (only declared fields allowed).
- LLM is invoked via `generate_content(model=..., contents=...)` in `orchestrator_node`.
- Always call `init_state_defaults(ctx)` at the start of every agent node.
- Use `ctx.route = next_agent` to set the graph routing at the end of `orchestrator_node`.

## Testing
- Integration tests require `GROQ_API_KEY` and mock server running on port 8001.
- Run a single test: `python -m unittest test_orchestrator.TestVoiceAgentOrchestrator.test_scenario_a_happy_path`
- CI only runs `test_session_state.py` (no LLM key needed).

## Do Not
- Do not add fields to `ctx.state` without declaring them in `session_state.py`.
- Do not route to `OfferAgent` before identity is verified (`current_agent` must have passed `GreetingAgent` → `VerificationAgent` → `EventAgent`).
- Do not mention competitor brand names (Zara, Lifestyle, H&M etc.) in any agent response.
- Do not terminate the call on the first prompt injection — warn and deflect first.
