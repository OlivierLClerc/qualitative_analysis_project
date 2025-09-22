#!/usr/bin/env python3
"""
Test script to verify that the iterative prompt improvement functionality works correctly.

This script tests both:
1. Single iteration mode (max_iterations = 1) - backward compatibility
2. Multi-iteration mode (max_iterations > 1) - new functionality
"""

import pandas as pd
from qualitative_analysis.scenario_runner import run_scenarios


def create_test_data():
    """Create a small test dataset for validation."""
    test_data = pd.DataFrame(
        {
            "verbatim": [
                "réponse_attendue: The answer is 42\n\nréponse_llm: The answer is 42\n\n",
                "réponse_attendue: The answer is 42\n\nréponse_llm: I don't know\n\n",
                "réponse_attendue: The answer is 42\n\nréponse_llm: The answer is 43\n\n",
                "réponse_attendue: The answer is 42\n\nréponse_llm: The answer is forty-two\n\n",
                "réponse_attendue: The answer is 42\n\nréponse_llm: 42 is the answer\n\n",
            ],
            "Rater_Oli": [1, 0, 0, 1, 1],
            "Rater_chloe": [1, 0, 0, 1, 1],
            "Rater_RA": [1, 0, 0, 1, 1],
            "iteration": [1, 1, 1, 1, 1],
        }
    )
    return test_data


def test_single_iteration():
    """Test single iteration mode (backward compatibility)."""
    print("=== Testing Single Iteration Mode (Backward Compatibility) ===")

    # Create test scenario with max_iterations = 1 (or not specified)
    single_iteration_scenario = [
        {
            # LLM settings
            "provider_llm1": "azure",
            "model_name_llm1": "gpt-4o",
            "temperature_llm1": 0,
            "prompt_name": "single_iteration_test",
            "subsample_size": -1,
            # Prompt
            "template": """
You are evaluating data entries.

Data columns:
- "réponse_attendue": Expected answer
- "réponse_llm": LLM response to evaluate

Entry to evaluate:
{verbatim_text}

Task: Evaluate if réponse_llm matches réponse_attendue.
0: Does not match
1: Matches
""",
            # Output
            "selected_fields": ["Classification", "Reasoning"],
            "prefix": "Classification",
            "label_type": "int",
            "response_template": """
Please follow the JSON format:
```json
{{
  "Reasoning": "Your reasoning here",
  "Classification": "Your integer here"
}}
""",
            "json_output": True,
            # Single iteration settings
            "max_iterations": 1,  # This should use process_scenario_raw
            "use_validation_set": False,
            "n_completions": 1,
        }
    ]

    test_data = create_test_data()
    annotation_columns = ["Rater_Oli", "Rater_chloe", "Rater_RA"]
    labels = [0, 1]

    try:
        results = run_scenarios(
            scenarios=single_iteration_scenario,
            data=test_data,
            annotation_columns=annotation_columns,
            labels=labels,
            n_runs=1,
            verbose=True,
        )

        print("✅ Single iteration test passed!")
        print(f"   - Results shape: {results.shape}")
        print(f"   - Columns: {list(results.columns)}")
        print(f"   - Iteration values: {results['iteration'].unique()}")

        # Check that iteration column shows 1 (single iteration)
        assert all(
            results["iteration"] == 1
        ), "Single iteration should have iteration=1"

        return True

    except Exception as e:
        print(f"❌ Single iteration test failed: {e}")
        return False


def test_multi_iteration():
    """Test multi-iteration mode (new functionality)."""
    print("\n=== Testing Multi-Iteration Mode (New Functionality) ===")

    # Create test scenario with max_iterations > 1
    multi_iteration_scenario = [
        {
            # LLM settings
            "provider_llm1": "azure",
            "model_name_llm1": "gpt-4o",
            "temperature_llm1": 0,
            "prompt_name": "multi_iteration_test",
            "subsample_size": -1,
            # Prompt
            "template": """
You are evaluating data entries.

Data columns:
- "réponse_attendue": Expected answer
- "réponse_llm": LLM response to evaluate

Entry to evaluate:
{verbatim_text}

Task: Evaluate if réponse_llm matches réponse_attendue.
0: Does not match
1: Matches
""",
            # Output
            "selected_fields": ["Classification", "Reasoning"],
            "prefix": "Classification",
            "label_type": "int",
            "response_template": """
Please follow the JSON format:
```json
{{
  "Reasoning": "Your reasoning here",
  "Classification": "Your integer here"
}}
""",
            "json_output": True,
            # Multi-iteration settings
            "provider_llm2": "azure",
            "model_name_llm2": "gpt-4o",
            "temperature_llm2": 0.7,
            "max_iterations": 2,  # This should use process_scenario
            "use_validation_set": True,
            "validation_size": 2,
            "random_state": 42,
            "n_completions": 1,
        }
    ]

    test_data = create_test_data()
    annotation_columns = ["Rater_Oli", "Rater_chloe", "Rater_RA"]
    labels = [0, 1]

    try:
        results = run_scenarios(
            scenarios=multi_iteration_scenario,
            data=test_data,
            annotation_columns=annotation_columns,
            labels=labels,
            n_runs=1,
            verbose=True,
        )

        print("✅ Multi-iteration test passed!")
        print(f"   - Results shape: {results.shape}")
        print(f"   - Columns: {list(results.columns)}")
        print(f"   - Iteration values: {results['iteration'].unique()}")

        # Check that iteration column shows max_iterations (2)
        assert all(
            results["iteration"] == 2
        ), "Multi-iteration should have iteration=max_iterations"

        # Check for additional columns that should be present in multi-iteration mode
        expected_columns = ["best_accuracy", "total_iterations"]
        for col in expected_columns:
            assert col in results.columns, f"Multi-iteration should have {col} column"

        return True

    except Exception as e:
        print(f"❌ Multi-iteration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Iterative Prompt Improvement Functionality")
    print("=" * 60)

    # Test single iteration (backward compatibility)
    single_success = test_single_iteration()

    # Test multi-iteration (new functionality)
    multi_success = test_multi_iteration()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Single iteration test: {'✅ PASSED' if single_success else '❌ FAILED'}")
    print(f"Multi-iteration test: {'✅ PASSED' if multi_success else '❌ FAILED'}")

    if single_success and multi_success:
        print(
            "\n🎉 All tests passed! The iterative functionality is working correctly."
        )
        print("\n" + "=" * 60)
        print("HOW THE SYSTEM WORKS:")
        print("=" * 60)
        print("\n📋 SINGLE ITERATION (max_iterations = 1):")
        print("   For each scenario:")
        print("   ├── Run 1: Original prompt → predictions")
        print("   ├── Run 2: Original prompt → predictions")
        print("   └── Run N: Original prompt → predictions")
        print("   → All runs use the SAME prompt for proper aggregation")

        print("\n🔄 MULTI-ITERATION (max_iterations > 1):")
        print("   For each scenario:")
        print("   ├── ITERATIVE IMPROVEMENT PHASE:")
        print("   │   ├── Iteration 1: Original prompt → accuracy_1")
        print("   │   ├── Iteration 2: LLM2 improves prompt → accuracy_2")
        print("   │   └── Iteration N: LLM2 improves prompt → accuracy_N")
        print("   │   → Select BEST prompt based on validation accuracy")
        print("   │")
        print("   └── CONSISTENCY MEASUREMENT PHASE:")
        print("       ├── Run 1: Best prompt → predictions")
        print("       ├── Run 2: Best prompt → predictions")
        print("       └── Run N: Best prompt → predictions")
        print("       → All runs use the SAME best prompt for proper aggregation")

        print("\n✅ BENEFITS:")
        print("   • Single iteration: Multiple runs with same prompt = proper metrics")
        print("   • Multi-iteration: Find best prompt once, then measure consistency")
        print("   • Metrics can be properly aggregated across runs")
        print("   • Perfect backward compatibility")

    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

    return single_success and multi_success


if __name__ == "__main__":
    main()
