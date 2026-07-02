import os
import json
import pytest
import asyncio
from orchestrator import classify_turn

# Load labeled dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "classification_dataset.json")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    CLASSIFICATION_TEST_CASES = json.load(f)

@pytest.mark.asyncio
@pytest.mark.parametrize("case", CLASSIFICATION_TEST_CASES)
async def test_classify_turn_regression(case):
    """
    Data-driven regression test for classify_turn().
    Runs live API queries against Groq, checking intent flags and confidence score thresholds.
    """
    user_input = case["user_input"]
    state = case["state"]
    expected = case["expected"]

    # Sleep to stay within Groq TPM limit (6000 TPM)
    await asyncio.sleep(12.0)

    # Invoke isolated classification function
    result = await classify_turn(user_input, state)

    print(f"\nTested: '{user_input}' (current_agent: {state.get('current_agent')})")
    print(f"Result: {result.model_dump()}")

    # Verify expected parameters
    for key, val in expected.items():
        if key == "confidence_score_lt":
            assert result.confidence_score < val, f"Expected confidence < {val}, got {result.confidence_score}"
        elif key == "confidence_score_gte":
            assert result.confidence_score >= val, f"Expected confidence >= {val}, got {result.confidence_score}"
        else:
            actual_val = getattr(result, key)
            assert actual_val == val, f"Expected {key} to be {val}, got {actual_val}"
