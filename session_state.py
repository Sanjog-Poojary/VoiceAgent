# session_state.py
# Kept for backwards compatibility.
# SessionState is now defined in orchestrator.py as part of the
# single-agent tool-calling architecture migration (July 2026).
try:
    from orchestrator import SessionState
except ImportError:
    from VoiceAgent.orchestrator import SessionState

__all__ = ["SessionState"]
