import re

with open('orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove confidence from Critique class definition
content = re.sub(r'    confidence: float = Field\(default=1\.0, ge=0\.0, le=1\.0, description="How certain the critic is about its own assessment\."\)\n', '', content)

# 2. Remove confidence=... from all Critique(...) instantiations
content = re.sub(r',\s*confidence=[0-9.]+', '', content)

# 3. Remove should_revise from AgentContract
# We will do this by regex or string replacement
should_revise_code = '''    def should_revise(self, critique: Critique, state: dict) -> bool:
        """Policy gate: determines whether a critique is strong enough to trigger revision.
        Default: revise if critique is unacceptable, critic confidence >= 0.7,
        and revision_count < 1 (consecutive-turn cap)."""
        if critique.is_acceptable:
            return False
        if state.get("revision_count", 0) >= 1:
            return False
        if critique.confidence < 0.7:
            return False
        return True'''
content = content.replace(should_revise_code, "")

# 4. Inline should_revise in _apply_critic_pass
apply_critic_pass_old = '''    if not critique.is_acceptable and contract.should_revise(critique, state):'''
apply_critic_pass_new = '''    if not critique.is_acceptable and state.get("revision_count", 0) < 1:'''
content = content.replace(apply_critic_pass_old, apply_critic_pass_new)

# 5. Remove redundant reflection fields from _apply_critic_pass
reflection_old = '''        ctx.state["revision_count"] = new_rev_count
        ctx.state["revision_reason"] = new_rev_reason
        ctx.state["reflection_status"] = refl_status
        ctx.state["revision_applied"] = rev_applied
        ctx.state["last_critique"] = ""  # cleared each turn; set by debug logging if needed'''
reflection_new = '''        ctx.state["revision_count"] = new_rev_count
        ctx.state["revision_reason"] = new_rev_reason'''
content = content.replace(reflection_old, reflection_new)

reflection_old2 = '''        ctx.state["last_decision"] = next_agent
        ctx.state["last_decision_confidence"] = classification.confidence_score'''
content = content.replace(reflection_old2, "")

# 6. Extract PlanningAgentContract and fix AgentContract.determine_next_agent
agent_contract_determine_next = '''    def determine_next_agent(self, classification: TurnClassification, state: dict, user_input_str: str) -> tuple[str, dict]:
        memory = state.get("agent_memory", {})
        updates = {}
        
        # Global Tangent Recovery & Guardrails
        plans = state.get("bounded_plans", {})
        for agent_name, plan in plans.items():
            if agent_name != self.name and getattr(plan, "plan_status", plan.get("plan_status")) == "In Progress":
                if state.get("last_outcome") == "tangent" or self.goal_satisfied(classification, memory, state):
                    rev_count = getattr(plan, "revision_count", plan.get("revision_count", 0))
                    max_revs = getattr(plan, "max_revisions", plan.get("max_revisions", 3))
                    
                    if rev_count >= max_revs:
                        if isinstance(plan, dict):
                            plan["plan_status"] = "Abandoned"
                        else:
                            plan.plan_status = "Abandoned"
                        updates["bounded_plans"] = plans
                        return "ApologyAgent", updates
                    
                    if state.get("last_outcome") == "tangent":
                        if isinstance(plan, dict):
                            plan["revision_count"] = rev_count + 1
                        else:
                            plan.revision_count = rev_count + 1
                        updates["bounded_plans"] = plans
                    else:
                        if isinstance(plan, dict):
                            plan["is_resuming"] = True
                        else:
                            plan.is_resuming = True
                        updates["bounded_plans"] = plans
                        return agent_name, updates

        if self.goal_satisfied(classification, memory, state):
            next_agent, route_updates = self._route_on_goal_complete(state)
        else:
            next_agent, route_updates = self._route_on_goal_incomplete(classification, state, user_input_str)
            
        updates.update(route_updates)
        return next_agent, updates'''

new_agent_contract_determine_next = '''    def determine_next_agent(self, classification: TurnClassification, state: dict, user_input_str: str) -> tuple[str, dict]:
        memory = state.get("agent_memory", {})
        if self.goal_satisfied(classification, memory, state):
            return self._route_on_goal_complete(state)
        return self._route_on_goal_incomplete(classification, state, user_input_str)

class PlanningAgentContract(AgentContract):
    def determine_next_agent(self, classification: TurnClassification, state: dict, user_input_str: str) -> tuple[str, dict]:
        memory = state.get("agent_memory", {})
        updates = {}
        
        # Global Tangent Recovery & Guardrails
        plans = state.get("bounded_plans", {})
        for agent_name, plan in plans.items():
            plan_status = getattr(plan, "plan_status", plan.get("plan_status", "")) if isinstance(plan, dict) else getattr(plan, "plan_status", "")
            if agent_name != self.name and plan_status == "In Progress":
                if state.get("last_outcome") == "tangent" or self.goal_satisfied(classification, memory, state):
                    rev_count = plan.get("revision_count", 0) if isinstance(plan, dict) else getattr(plan, "revision_count", 0)
                    max_revs = plan.get("max_revisions", 3) if isinstance(plan, dict) else getattr(plan, "max_revisions", 3)
                    
                    if rev_count >= max_revs:
                        if isinstance(plan, dict):
                            plan["plan_status"] = "Abandoned"
                        else:
                            plan.plan_status = "Abandoned"
                        updates["bounded_plans"] = plans
                        return "ApologyAgent", updates
                    
                    if state.get("last_outcome") == "tangent":
                        if isinstance(plan, dict):
                            plan["revision_count"] = rev_count + 1
                        else:
                            plan.revision_count = rev_count + 1
                        updates["bounded_plans"] = plans
                    else:
                        if isinstance(plan, dict):
                            plan["is_resuming"] = True
                        else:
                            plan.is_resuming = True
                        updates["bounded_plans"] = plans
                        return agent_name, updates

        if self.goal_satisfied(classification, memory, state):
            next_agent, route_updates = self._route_on_goal_complete(state)
        else:
            next_agent, route_updates = self._route_on_goal_incomplete(classification, state, user_input_str)
            
        updates.update(route_updates)
        return next_agent, updates'''

content = content.replace(agent_contract_determine_next, new_agent_contract_determine_next)

# 7. Update subclasses to inherit from PlanningAgentContract
content = content.replace("class VerificationAgentContract(AgentContract):", "class VerificationAgentContract(PlanningAgentContract):")
content = content.replace("class SpendingHistoryAgentContract(AgentContract):", "class SpendingHistoryAgentContract(PlanningAgentContract):")
content = content.replace("class OfferAgentContract(AgentContract):", "class OfferAgentContract(PlanningAgentContract):")

with open('orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactoring complete.")
