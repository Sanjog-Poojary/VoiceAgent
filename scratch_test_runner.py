import asyncio
from orchestrator import VoiceAgentWorkflow
from google.adk.runners import Runner
from google.adk.apps import App
from google.genai import types

async def main():
    from mock_server import make_resume_message
    app = App(name='t', root_agent=VoiceAgentWorkflow(name='w'))
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    runner = Runner(app=app, auto_create_session=True, session_service=InMemorySessionService())
    
    print("--- TURN 1 ---")
    msg1 = types.Content(role="user", parts=[types.Part(text="[Call Connected]")])
    int_id = None
    async for e in runner.run_async(user_id='1', session_id='1', new_message=msg1):
        if e.output: print("Output:", e.output)
        if getattr(e, "content", None) and e.content.parts:
            part = e.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_input":
                int_id = part.function_call.id
                print("Yielded message:", part.function_call.args.get("message"))
                
    print("--- TURN 2 ---")
    msg2 = make_resume_message(int_id, "yes it is")
    async for e in runner.run_async(user_id='1', session_id='1', new_message=msg2):
        if getattr(e, "content", None) and e.content.parts:
            part = e.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_input":
                int_id = part.function_call.id
                print("Yielded message:", part.function_call.args.get("message"))

    print("--- TURN 3 ---")
    msg3 = make_resume_message(int_id, "what is it")
    async for e in runner.run_async(user_id='1', session_id='1', new_message=msg3):
        if getattr(e, "content", None) and e.content.parts:
            part = e.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_input":
                int_id = part.function_call.id
                print("Yielded message:", part.function_call.args.get("message"))

    print("--- TURN 4 ---")
    msg4 = make_resume_message(int_id, "ohh")
    async for e in runner.run_async(user_id='1', session_id='1', new_message=msg4):
        if getattr(e, "content", None) and e.content.parts:
            part = e.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_input":
                int_id = part.function_call.id
                print("Yielded message:", part.function_call.args.get("message"))

    print("--- TURN 5 ---")
    msg5 = make_resume_message(int_id, "yes")
    async for e in runner.run_async(user_id='1', session_id='1', new_message=msg5):
        if getattr(e, "content", None) and e.content.parts:
            part = e.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_input":
                int_id = part.function_call.id
                print("Yielded message:", part.function_call.args.get("message"))
                
    print("--- TURN 6 ---")
    msg6 = make_resume_message(int_id, "yes I would like to hear the offer")
    async for e in runner.run_async(user_id='1', session_id='1', new_message=msg6):
        if getattr(e, "content", None) and e.content.parts:
            part = e.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_input":
                int_id = part.function_call.id
                print("Yielded message:", part.function_call.args.get("message"))

asyncio.run(main())
