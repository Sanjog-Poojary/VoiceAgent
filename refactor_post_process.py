import re

with open('orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix OfferAgentContract.post_process
old_offer = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("OfferAgent")
        if not plan or plan.get("plan_status") != "In Progress":'''

new_offer = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("OfferAgent")
        plan_status = plan.get("plan_status") if isinstance(plan, dict) else getattr(plan, "plan_status", "") if plan else ""
        if not plan or plan_status != "In Progress":'''
content = content.replace(old_offer, new_offer)

# Fix VerificationAgentContract.post_process
old_verif = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("VerificationAgent")
        if not plan or plan.get("plan_status") != "In Progress":'''

new_verif = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("VerificationAgent")
        plan_status = plan.get("plan_status") if isinstance(plan, dict) else getattr(plan, "plan_status", "") if plan else ""
        if not plan or plan_status != "In Progress":'''
content = content.replace(old_verif, new_verif)

# Also fix the assignment part in VerificationAgentContract where it does plan["plan_status"] etc.
# Actually, Pydantic objects need attribute access.
# I will use a helper to set attributes to handle both dict and Pydantic safely.
# Wait, in the orchestrator, plans inside `state.setdefault("bounded_plans", {})` are Pydantic objects if state is from ADK, but if we assign a dict, ADK merges it.
# Let's just fix the `plan.get("plan_status")` part first.

# Since we want a foolproof way to mutate plan:
def dict_or_attr_replacer(match):
    return match.group(1) + '''
        if isinstance(plan, dict):
            plan["active_step"] = "Confirm Acceptance"
        else:
            plan.active_step = "Confirm Acceptance"
'''

# We will just write a python script to fix the `plan["..."] = ...` and `.append()` calls.
# Wait, let's just make `plan` a dict if it is a Pydantic object!
# plan = plan.model_dump() if hasattr(plan, "model_dump") else plan
# Then we can mutate the dict, and reassign it: plans["OfferAgent"] = plan

old_offer_full = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("OfferAgent")
        if not plan or plan.get("plan_status") != "In Progress":'''

new_offer_full = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("OfferAgent")
        if hasattr(plan, "model_dump"): plan = plan.model_dump()
        if not plan or plan.get("plan_status") != "In Progress":'''

old_verif_full = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("VerificationAgent")
        if not plan or plan.get("plan_status") != "In Progress":'''

new_verif_full = '''    async def post_process(self, classification, memory, state):
        plans = state.setdefault("bounded_plans", {})
        plan = plans.get("VerificationAgent")
        if hasattr(plan, "model_dump"): plan = plan.model_dump()
        if not plan or plan.get("plan_status") != "In Progress":'''

content = content.replace(old_offer, new_offer_full)
content = content.replace(old_verif, new_verif_full)

# But wait! If we do `plan = plan.model_dump()`, we mutate a local dict. We MUST reassign it back to `plans["AgentName"]` at the end!
# Actually, if we reassign `plans["OfferAgent"] = plan` unconditionally at the end of post_process...
# Let's do that!

# For OfferAgent:
offer_return = '''        return last_outcome, memory'''
new_offer_return = '''        if plan: plans["OfferAgent"] = plan
        return last_outcome, memory'''
content = content.replace(offer_return, new_offer_return, 1)

# For VerificationAgent:
verif_return = '''        return last_outcome, memory'''
new_verif_return = '''        if plan: plans["VerificationAgent"] = plan
        return last_outcome, memory'''
content = content.replace(verif_return, new_verif_return, 1)

with open('orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("post_process refactored.")
