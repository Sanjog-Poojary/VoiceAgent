import asyncio
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.evaluation.eval_config import EvalConfig, EvaluationCriteria
from google.adk.evaluation.eval_set import EvalSet
import json

async def main():
    service = LocalEvalService()
    
    # Load golden evalset
    with open('voice_agent_eval/golden.evalset.json', 'r') as f:
        data = json.load(f)
        
    # Just run the first one for testing
    cases = data['eval_cases'][:1]
    
    # Run evaluation
    config = EvalConfig(criteria=EvaluationCriteria())
    result = await service.run_eval_set(
        agent_module_file_path="agent.py",
        eval_set=EvalSet(eval_set_id="test", eval_cases=cases),
        config=config
    )
    print(result)

asyncio.run(main())
