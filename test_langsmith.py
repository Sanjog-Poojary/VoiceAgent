import asyncio
import os
from orchestrator import classify_turn, PlanningAgentContract, TurnClassification, AgentContract

async def main():
    # Force the key if it's not set so we get a trace
    if not os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGCHAIN_API_KEY") == "your_langsmith_api_key_here":
        print("Warning: LANGCHAIN_API_KEY is not set correctly in environment, trace may fail to upload.")

    state = {
        "current_agent": "SpendingHistoryAgent",
        "offer_pitched": True,
        "bounded_plans": {}
    }
    user_input = "I'm busy right now, can I book a personal shopper for later"
    
    print(f"Running deliberately broken test case: {user_input!r}")
    print("Step 1: Classifying turn...")
    classification = await classify_turn(user_input, state)
    print("Classification result:")
    print(classification.model_dump_json(indent=2))
    
    # We use a mocked contract that resembles SpendingHistoryAgent
    contract = PlanningAgentContract(
        name="SpendingHistoryAgent", 
        goal="retrieve_spending_history_and_pitch_interest", 
        expected_input="Customer response showing interest", 
        success_criteria="Spending history context shared", 
        possible_next_actions=["OfferAgent", "PersonalShopperAgent", "ApologyAgent"]
    )
    
    print("\nStep 2: Determining Next Agent...")
    next_agent, updates = contract.determine_next_agent(classification, state, user_input)
    print("Proposed Next Agent:", next_agent)
    
    print("\nStep 3: Criticize Decision...")
    critique = contract.criticize_decision(classification, state, next_agent, updates, user_input)
    print("Critique acceptable?", critique.is_acceptable)
    
    if not critique.is_acceptable:
        print("\nStep 4: Revise Decision...")
        next_agent, updates = contract.revise_decision(classification, state, critique, next_agent, updates, user_input)
        print("Revised Next Agent:", next_agent)

if __name__ == "__main__":
    asyncio.run(main())
