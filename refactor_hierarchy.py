with open('orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to find `class PlanningAgentContract(AgentContract):`
# and move it to AFTER `revise_decision` in `AgentContract`.

# Currently:
# class PlanningAgentContract(AgentContract):
#     def determine_next_agent(...)
#         ...
#     def _route_on_goal_complete(...)
#     def _route_on_goal_incomplete(...)
#     def criticize_decision(...)
#     def revise_decision(...)

# The easiest way is to remove `class PlanningAgentContract(AgentContract):` and `def determine_next_agent(...)`
# and insert them AFTER `revise_decision`.

# Let's extract the exact text of PlanningAgentContract.determine_next_agent

start_idx = content.find("class PlanningAgentContract(AgentContract):")
if start_idx != -1:
    end_idx = content.find("    def _route_on_goal_complete(self, state: dict) -> tuple[str, dict]:", start_idx)
    planning_agent_code = content[start_idx:end_idx]
    
    # Remove planning_agent_code from its current location
    content = content[:start_idx] + content[end_idx:]
    
    # Now, find the end of AgentContract (which is before `class IdentityConfirmationContract`)
    insert_idx = content.find("class IdentityConfirmationContract")
    
    # Insert planning_agent_code there
    content = content[:insert_idx] + planning_agent_code + "\n" + content[insert_idx:]
    
    with open('orchestrator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed AgentContract hierarchy!")
else:
    print("Could not find PlanningAgentContract.")
