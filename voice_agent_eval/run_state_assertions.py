"""
run_state_assertions.py
"""

from __future__ import annotations

import json
import asyncio
import sys
import subprocess
from pathlib import Path
from typing import Any

import pytest

# Adjust imports to read from local modules
try:
    from orchestrator import VoiceAgentWorkflow
except ModuleNotFoundError:
    from VoiceAgent.orchestrator import VoiceAgentWorkflow

from test_orchestrator import run_turn, make_user_message, make_resume_message

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.apps import App, ResumabilityConfig

GOLDEN_PATH = Path(__file__).parent / "golden.evalset.json"


def _load_golden_cases() -> list[dict[str, Any]]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["eval_cases"]


def _user_texts(conversation: list[dict[str, Any]]) -> list[str]:
    texts = []
    for inv in conversation:
        parts = inv["userContent"]["parts"]
        texts.append(" ".join(p.get("text", "") for p in parts))
    return texts


GOLDEN_CASES = _load_golden_cases()
CASE_IDS = [c["evalId"] for c in GOLDEN_CASES]


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module", autouse=True)
def mock_server():
    python_exe = sys.executable
    server_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "mock_server:app", "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    # Wait for the mock server to start
    import time
    time.sleep(3.0)
    yield
    server_process.terminate()
    server_process.wait()
    if server_process.stdout:
        server_process.stdout.close()
    if server_process.stderr:
        server_process.stderr.close()


def build_runner():
    session_service = InMemorySessionService()
    workflow = VoiceAgentWorkflow(name="voice_agent_workflow")
    app = App(
        name="VoiceAgent", root_agent=workflow,
        resumability_config=ResumabilityConfig(is_resumable=True)
    )
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)
    return runner, session_service


@pytest.mark.asyncio
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
async def test_golden_case_final_state(case: dict[str, Any]):
    """Drives every turn in a golden case through the real orchestrator
    graph and asserts the resulting session state matches finalSessionState.
    """
    expected_state: dict[str, Any] = case.get("finalSessionState") or {}
    if not expected_state:
        pytest.skip(f"{case['evalId']}: no finalSessionState to assert")

    runner, session_service = build_runner()
    session_id = "eval_" + case["evalId"]
    user_id = "eval_user"

    # Make initial state delta
    customer_id = "1"
    if "suspicious" in case["evalId"] or "sarcastic" in case["evalId"]:
        customer_id = "2"
    
    initial_state = {
        "customer_id": customer_id,
        "detected_language": "English",
        "current_agent": "GreetingAgent",
        "verification_attempts": 0,
        "call_sentiment": "Neutral",
        "offer_pitched": False,
        "offer_accepted": False,
        "escalation_triggered": False,
        "raw_audio_transcription": []
    }

    # Run first turn using initial state delta
    user_texts = _user_texts(case["conversation"])
    
    # Sleep to avoid Groq rate limit between test cases
    await asyncio.sleep(8.0)
    
    # We always need to trigger initial turn
    agent_message, interrupt_id, invocation_id = await run_turn(
        runner, user_id, session_id,
        make_user_message("[Call Connected]"), state_delta=initial_state
    )

    for user_text in user_texts:
        await asyncio.sleep(8.0) # rate limiting
        agent_message, interrupt_id, invocation_id = await run_turn(
            runner=runner,
            user_id=user_id,
            session_id=session_id,
            message=make_resume_message(interrupt_id, user_text),
            invocation_id=invocation_id
        )

    # Get final state
    session = await session_service.get_session(
        app_name="VoiceAgent", user_id=user_id, session_id=session_id
    )
    state = session.state if session else {}

    mismatches = {
        key: (expected_val, state.get(key, "<missing>"))
        for key, expected_val in expected_state.items()
        if state.get(key, "<missing>") != expected_val
    }

    assert not mismatches, (
        f"{case['evalId']}: state mismatch after final turn -> "
        + ", ".join(f"{k}: expected={v[0]!r} actual={v[1]!r}" for k, v in mismatches.items())
    )