from google.adk.apps import App, ResumabilityConfig
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
