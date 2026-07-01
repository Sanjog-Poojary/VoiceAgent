from google.adk.apps import App, ResumabilityConfig
import sys
try:
    from orchestrator import VoiceAgentWorkflow
except ModuleNotFoundError:
    from VoiceAgent.orchestrator import VoiceAgentWorkflow

root_agent = VoiceAgentWorkflow(name="voice_agent_workflow")
app = App(
    name="VoiceAgent",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True)
)

# adk web/eval loads modules dynamically using different module names (e.g. "VoiceAgent.agent" or "agent.agent").
# Dynamically add the self-reference attribute on whatever module name we were loaded under.
if __name__ in sys.modules:
    sys.modules[__name__].agent = sys.modules[__name__]
