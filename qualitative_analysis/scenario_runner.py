"""
scenario_runner.py

This module provides functions for running multiple scenarios with language models,
computing metrics, and aggregating results across multiple runs. It's designed to
support both notebook and application workflows for qualitative analysis tasks.

Dependencies:
    - pandas
    - numpy
    - sklearn.model_selection (train_test_split)
    - qualitative_analysis.prompt_engineering (run_iterative_prompt_improvement)
    - qualitative_analysis.alt_test (benjamini_yekutieli_correction)
    - qualitative_analysis.evaluation (compute_classification_metrics, compute_all_kappas)

Functions:
    - process_scenario(scenario, labeled_data, annotation_columns, labels, epsilon, verbose):
        Processes a single scenario with the specified parameters.
        
    - run_scenarios(scenarios, labeled_data, annotation_columns, labels, n_runs, verbose):
        Runs multiple scenarios and returns aggregated results.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple
from qualitative_analysis.prompt_engineering import run_iterative_prompt_improvement
from qualitative_analysis.model_interaction import get_llm_client
from qualitative_analysis.notebooks_functions import process_general_verbatims
import qualitative_analysis.config as config


def process_scenario(
    scenario: Dict[str, Any],
    labeled_data: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    epsilon: float = 0.2,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process a single scenario with the specified parameters.

    Parameters
    ----------
    scenario : Dict[str, Any]
        Dictionary containing scenario configuration parameters.
    labeled_data : pd.DataFrame
        DataFrame containing labeled data.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    verbose : bool, optional
        Whether to print verbose output, by default True

    Returns
    -------
    Tuple[List[Dict[str, Any]], Dict[str, Any]]
        A tuple containing:
        - A list of dictionaries with results from each run
        - A dictionary with the best prompt and accuracy
    """
    # Extract data configuration from scenario
    subsample_size = scenario.get("subsample_size", 20)
    use_validation_set = scenario.get("use_validation_set", True)
    validation_size = scenario.get("validation_size", 10)
    random_state = scenario.get("random_state", 42)

    # Step 1: Get a stratified subset of samples or use full dataset
    if subsample_size == "all" or subsample_size == -1 or subsample_size == "annotated":
        # Use all labeled data
        data_subset = labeled_data
        print(f"Using all labeled data: {len(data_subset)} samples")
    else:
        # Use a subset as before
        data_subset, _ = train_test_split(
            labeled_data,
            train_size=subsample_size,
            # stratify=labeled_data['label'] if 'label' in labeled_data.columns else None,
            random_state=random_state,
        )

    # Step 2: Split subset into train/val if use_validation_set is True
    if use_validation_set:
        train_data, val_data = train_test_split(
            data_subset,
            test_size=validation_size,
            # stratify=data_subset['label'] if 'label' in data_subset.columns else None,
            random_state=random_state,
        )
        print(
            f"Scenario '{scenario['prompt_name']}' - Train size: {len(train_data)}, Val size: {len(val_data)}"
        )
    else:
        # Use all data for training
        train_data = data_subset
        val_data = None  # No validation set
        print(
            f"Scenario '{scenario['prompt_name']}' - Train size (all data): {len(train_data)}, No validation set"
        )

    best_prompt_overall = None
    best_accuracy_overall = -1.0  # Using float to avoid type error

    # Run the iterative prompt improvement
    best_prompt, best_accuracy, iteration_rows = run_iterative_prompt_improvement(
        scenario=scenario,
        train_data=train_data,
        val_data=val_data,  # This can now be None
        annotation_columns=annotation_columns,
        labels=labels,
        alt_test=True,
        errors_examples=0.5,
        examples_to_give=4,
        epsilon=epsilon,
        verbose=verbose,
    )

    # Track the best prompt
    if best_accuracy > best_accuracy_overall:
        best_accuracy_overall = best_accuracy
        best_prompt_overall = best_prompt

    # Return the results
    return iteration_rows, {
        "best_prompt": best_prompt_overall,
        "best_accuracy": best_accuracy_overall,
    }


def process_scenario_raw(
    scenario: Dict[str, Any],
    data: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a single scenario and return only the raw data with model predictions and metadata.

    Parameters
    ----------
    scenario : Dict[str, Any]
        Dictionary containing scenario configuration parameters.
    data : pd.DataFrame
        DataFrame containing labeled data.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    verbose : bool, optional
        Whether to print verbose output, by default True

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with raw data for each sample
    """
    # Extract data configuration from scenario
    subsample_size = scenario.get("subsample_size", 20)
    use_validation_set = scenario.get("use_validation_set", True)
    validation_size = scenario.get("validation_size", 10)
    random_state = scenario.get("random_state", 42)

    # Step 1: Get a stratified subset of samples or use full dataset
    if subsample_size == "all" or subsample_size == -1 or subsample_size == "annotated":
        # Use all labeled data
        data_subset = data
        print(f"Using all labeled data: {len(data_subset)} samples")
    else:
        # Use a subset as before
        data_subset, _ = train_test_split(
            data,
            train_size=subsample_size,
            # stratify=labeled_data['label'] if 'label' in labeled_data.columns else None,
            random_state=random_state,
        )

    # Step 2: Split subset into train/val if use_validation_set is True
    if use_validation_set:
        train_data, val_data = train_test_split(
            data_subset,
            test_size=validation_size,
            # stratify=data_subset['label'] if 'label' in data_subset.columns else None,
            random_state=random_state,
        )
        print(
            f"Scenario '{scenario['prompt_name']}' - Train size: {len(train_data)}, Val size: {len(val_data)}"
        )
    else:
        # Use all data for training
        train_data = data_subset
        val_data = None  # No validation set
        print(
            f"Scenario '{scenario['prompt_name']}' - Train size (all data): {len(train_data)}, No validation set"
        )

    # Get iteration from scenario or default to 1
    iteration = scenario.get("iteration", 1)

    # Get the prompt template from the scenario
    prompt_template = scenario["template"]

    # Get other parameters from the scenario
    provider_1 = scenario["provider_llm1"]
    model_name_1 = scenario["model_name_llm1"]
    temperature_llm1 = scenario["temperature_llm1"]
    prefix_llm1 = scenario.get("prefix", None)
    response_template = scenario.get("response_template", None)
    json_output = scenario.get("json_output", False)
    selected_fields = scenario.get("selected_fields", None)
    n_completions = scenario.get("n_completions", 1)

    # Initialize LLM client

    llm1_client = get_llm_client(
        provider=provider_1, config=config.MODEL_CONFIG[provider_1], model=model_name_1
    )

    # Prepare the full prompt
    if json_output and response_template:
        full_prompt = f"{prompt_template}\n\n{response_template}"
    else:
        full_prompt = prompt_template

    # Create raw data rows
    raw_data_rows = []

    # Process train data
    train_pred_df, train_cost_info, train_totals = process_general_verbatims(
        verbatims_subset=train_data["verbatim"].tolist(),
        llm_client=llm1_client,
        model_name=model_name_1,
        prompt_template=full_prompt,
        label_field=prefix_llm1,
        temperature=temperature_llm1,
        verbose=verbose,
        json_output=json_output,
        selected_fields=selected_fields,
        n_completions=n_completions,
    )

    # Add train data with real model predictions
    for i, row in train_data.iterrows():
        raw_row = {
            "sample_id": i,
            "split": "train",
            "verbatim": row["verbatim"],
            "iteration": iteration,  # Add iteration column
        }

        # Add human annotations
        for col in annotation_columns:
            if col in row:
                raw_row[col] = row[col]

        # Find the corresponding prediction in train_pred_df
        matching_pred = train_pred_df[train_pred_df["Verbatim"] == row["verbatim"]]
        if not matching_pred.empty:
            # First, set ModelPrediction based on selected_fields or Label
            if selected_fields and selected_fields[0] in matching_pred.columns:
                # Use the first selected field as ModelPrediction
                raw_row["ModelPrediction"] = matching_pred.iloc[0][selected_fields[0]]
            elif "Label" in matching_pred.columns:
                # Fall back to Label if no selected_fields or the field doesn't exist
                raw_row["ModelPrediction"] = matching_pred.iloc[0]["Label"]

            # Then add all other columns from the prediction to raw_row
            for col in matching_pred.columns:
                if col != "Verbatim":  # Skip the verbatim column
                    # Skip adding the column if it's already used as ModelPrediction
                    # to avoid redundancy
                    if (
                        not (selected_fields and col == selected_fields[0])
                        and col != "Label"
                    ):
                        raw_row[col] = matching_pred.iloc[0][col]
        else:
            # If no prediction found, use a default value
            raw_row["ModelPrediction"] = 1
            if verbose:
                print(f"Warning: No model prediction found for train sample {i}")

        raw_data_rows.append(raw_row)

    # Process validation data if available
    if val_data is not None:
        val_pred_df, val_cost_info, val_totals = process_general_verbatims(
            verbatims_subset=val_data["verbatim"].tolist(),
            llm_client=llm1_client,
            model_name=model_name_1,
            prompt_template=full_prompt,
            label_field=prefix_llm1,
            temperature=temperature_llm1,
            verbose=verbose,
            json_output=json_output,
            selected_fields=selected_fields,
            n_completions=n_completions,
        )

        # Add validation data with real model predictions
        for i, row in val_data.iterrows():
            raw_row = {
                "sample_id": i,
                "split": "val",
                "verbatim": row["verbatim"],
                "iteration": iteration,  # Add iteration column
            }

            # Add human annotations
            for col in annotation_columns:
                if col in row:
                    raw_row[col] = row[col]

            # Find the corresponding prediction in val_pred_df
            matching_pred = val_pred_df[val_pred_df["Verbatim"] == row["verbatim"]]
            if not matching_pred.empty:
                # First, set ModelPrediction based on selected_fields or Label
                if selected_fields and selected_fields[0] in matching_pred.columns:
                    # Use the first selected field as ModelPrediction
                    raw_row["ModelPrediction"] = matching_pred.iloc[0][
                        selected_fields[0]
                    ]
                elif "Label" in matching_pred.columns:
                    # Fall back to Label if no selected_fields or the field doesn't exist
                    raw_row["ModelPrediction"] = matching_pred.iloc[0]["Label"]

                # Then add all other columns from the prediction to raw_row
                for col in matching_pred.columns:
                    if col != "Verbatim":  # Skip the verbatim column
                        # Skip adding the column if it's already used as ModelPrediction
                        # to avoid redundancy
                        if (
                            not (selected_fields and col == selected_fields[0])
                            and col != "Label"
                        ):
                            raw_row[col] = matching_pred.iloc[0][col]
            else:
                # If no prediction found, use a default value
                raw_row["ModelPrediction"] = 1
                if verbose:
                    print(
                        f"Warning: No model prediction found for validation sample {i}"
                    )

            raw_data_rows.append(raw_row)

    return raw_data_rows


def process_scenario_with_final_prompt(
    scenario: Dict[str, Any],
    data: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    final_prompt: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a single scenario using a final optimized prompt and return raw data with model predictions.
    This is used after iterative prompt improvement to get predictions with the best prompt.

    Parameters
    ----------
    scenario : Dict[str, Any]
        Dictionary containing scenario configuration parameters.
    data : pd.DataFrame
        DataFrame containing labeled data.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    final_prompt : str
        The optimized prompt to use for predictions.
    verbose : bool, optional
        Whether to print verbose output, by default True

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with raw data for each sample
    """
    # Extract data configuration from scenario
    subsample_size = scenario.get("subsample_size", 20)
    use_validation_set = scenario.get("use_validation_set", True)
    validation_size = scenario.get("validation_size", 10)
    random_state = scenario.get("random_state", 42)

    # Step 1: Get a stratified subset of samples or use full dataset
    if subsample_size == "all" or subsample_size == -1 or subsample_size == "annotated":
        # Use all labeled data
        data_subset = data
        if verbose:
            print(f"Using all labeled data: {len(data_subset)} samples")
    else:
        # Use a subset as before
        data_subset, _ = train_test_split(
            data,
            train_size=subsample_size,
            random_state=random_state,
        )

    # Step 2: Split subset into train/val if use_validation_set is True
    if use_validation_set:
        train_data, val_data = train_test_split(
            data_subset,
            test_size=validation_size,
            random_state=random_state,
        )
        if verbose:
            print(
                f"Scenario '{scenario['prompt_name']}' - Train size: {len(train_data)}, Val size: {len(val_data)}"
            )
    else:
        # Use all data for training
        train_data = data_subset
        val_data = None  # No validation set
        if verbose:
            print(
                f"Scenario '{scenario['prompt_name']}' - Train size (all data): {len(train_data)}, No validation set"
            )

    # Get other parameters from the scenario
    provider_1 = scenario["provider_llm1"]
    model_name_1 = scenario["model_name_llm1"]
    temperature_llm1 = scenario["temperature_llm1"]
    prefix_llm1 = scenario.get("prefix", None)
    # response_template = scenario.get("response_template", None)
    json_output = scenario.get("json_output", False)
    selected_fields = scenario.get("selected_fields", None)
    n_completions = scenario.get("n_completions", 1)
    max_iterations = scenario.get("max_iterations", 1)

    # Initialize LLM client
    llm1_client = get_llm_client(
        provider=provider_1, config=config.MODEL_CONFIG[provider_1], model=model_name_1
    )

    # Use the final optimized prompt
    full_prompt = final_prompt

    # Create raw data rows
    raw_data_rows = []

    # Process train data
    train_pred_df, train_cost_info, train_totals = process_general_verbatims(
        verbatims_subset=train_data["verbatim"].tolist(),
        llm_client=llm1_client,
        model_name=model_name_1,
        prompt_template=full_prompt,
        label_field=prefix_llm1,
        temperature=temperature_llm1,
        verbose=verbose,
        json_output=json_output,
        selected_fields=selected_fields,
        n_completions=n_completions,
    )

    # Add train data with real model predictions
    for i, row in train_data.iterrows():
        raw_row = {
            "sample_id": i,
            "split": "train",
            "verbatim": row["verbatim"],
            "iteration": int(
                max_iterations
            ),  # Use max_iterations to indicate this used iterative improvement
        }

        # Add human annotations
        for col in annotation_columns:
            if col in row:
                raw_row[col] = row[col]

        # Find the corresponding prediction in train_pred_df
        matching_pred = train_pred_df[train_pred_df["Verbatim"] == row["verbatim"]]
        if not matching_pred.empty:
            # First, set ModelPrediction based on selected_fields or Label
            if selected_fields and selected_fields[0] in matching_pred.columns:
                # Use the first selected field as ModelPrediction
                raw_row["ModelPrediction"] = matching_pred.iloc[0][selected_fields[0]]
            elif "Label" in matching_pred.columns:
                # Fall back to Label if no selected_fields or the field doesn't exist
                raw_row["ModelPrediction"] = matching_pred.iloc[0]["Label"]

            # Then add all other columns from the prediction to raw_row
            for col in matching_pred.columns:
                if col != "Verbatim":  # Skip the verbatim column
                    # Skip adding the column if it's already used as ModelPrediction
                    # to avoid redundancy
                    if (
                        not (selected_fields and col == selected_fields[0])
                        and col != "Label"
                    ):
                        raw_row[col] = matching_pred.iloc[0][col]
        else:
            # If no prediction found, use a default value
            raw_row["ModelPrediction"] = 1
            if verbose:
                print(f"Warning: No model prediction found for train sample {i}")

        raw_data_rows.append(raw_row)

    # Process validation data if available
    if val_data is not None:
        val_pred_df, val_cost_info, val_totals = process_general_verbatims(
            verbatims_subset=val_data["verbatim"].tolist(),
            llm_client=llm1_client,
            model_name=model_name_1,
            prompt_template=full_prompt,
            label_field=prefix_llm1,
            temperature=temperature_llm1,
            verbose=verbose,
            json_output=json_output,
            selected_fields=selected_fields,
            n_completions=n_completions,
        )

        # Add validation data with real model predictions
        for i, row in val_data.iterrows():
            raw_row = {
                "sample_id": i,
                "split": "val",
                "verbatim": row["verbatim"],
                "iteration": int(
                    max_iterations
                ),  # Use max_iterations to indicate this used iterative improvement
            }

            # Add human annotations
            for col in annotation_columns:
                if col in row:
                    raw_row[col] = row[col]

            # Find the corresponding prediction in val_pred_df
            matching_pred = val_pred_df[val_pred_df["Verbatim"] == row["verbatim"]]
            if not matching_pred.empty:
                # First, set ModelPrediction based on selected_fields or Label
                if selected_fields and selected_fields[0] in matching_pred.columns:
                    # Use the first selected field as ModelPrediction
                    raw_row["ModelPrediction"] = matching_pred.iloc[0][
                        selected_fields[0]
                    ]
                elif "Label" in matching_pred.columns:
                    # Fall back to Label if no selected_fields or the field doesn't exist
                    raw_row["ModelPrediction"] = matching_pred.iloc[0]["Label"]

                # Then add all other columns from the prediction to raw_row
                for col in matching_pred.columns:
                    if col != "Verbatim":  # Skip the verbatim column
                        # Skip adding the column if it's already used as ModelPrediction
                        # to avoid redundancy
                        if (
                            not (selected_fields and col == selected_fields[0])
                            and col != "Label"
                        ):
                            raw_row[col] = matching_pred.iloc[0][col]
            else:
                # If no prediction found, use a default value
                raw_row["ModelPrediction"] = 1
                if verbose:
                    print(
                        f"Warning: No model prediction found for validation sample {i}"
                    )

            raw_data_rows.append(raw_row)

    return raw_data_rows


def run_scenarios(
    scenarios: List[Dict[str, Any]],
    data: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    n_runs: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run multiple scenarios and return the detailed results DataFrame.
    This function supports both single-iteration and iterative prompt improvement modes.

    - If max_iterations <= 1: Uses process_scenario_raw() for single iteration
    - If max_iterations > 1: Uses process_scenario() for iterative prompt improvement

    Parameters
    ----------
    scenarios : List[Dict[str, Any]]
        List of scenario dictionaries.
    data : pd.DataFrame
        DataFrame containing labeled data.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    n_runs : int, optional
        Number of runs per scenario, by default 2
    verbose : bool, optional
        Whether to print verbose output, by default True

    Returns
    -------
    pd.DataFrame
        A DataFrame with detailed results from all scenarios and runs
    """
    # For storing all individual run results
    all_detailed_results = []

    # Make a copy of labeled_data to avoid modifying the original
    data = data.copy()

    for scenario in scenarios:
        # Check if this scenario uses iterative prompt improvement
        max_iterations = scenario.get("max_iterations", 1)

        if max_iterations <= 1:
            # Single iteration mode - use the existing fast path
            for run in range(n_runs):
                raw_data_rows = process_scenario_raw(
                    scenario=scenario,
                    data=data,
                    annotation_columns=annotation_columns,
                    labels=labels,
                    verbose=verbose,
                )

                # Add run information to each row
                for row in raw_data_rows:
                    row["run"] = run + 1
                    row["prompt_name"] = scenario["prompt_name"]
                    row["use_validation_set"] = scenario.get("use_validation_set", True)
                    all_detailed_results.append(row)
        else:
            # Iterative prompt improvement mode
            if verbose:
                print(
                    f"\n=== Running iterative prompt improvement for scenario '{scenario['prompt_name']}' ==="
                )

            # Run the iterative improvement process once to get all iteration results
            iteration_rows, best_prompt_info = process_scenario(
                scenario=scenario,
                labeled_data=data,
                annotation_columns=annotation_columns,
                labels=labels,
                epsilon=0.2,
                verbose=verbose,
            )

            # Extract all prompts from the iteration process
            max_iterations = scenario.get("max_iterations", 1)

            # For each iteration, run it multiple times (n_runs)
            for iteration_num in range(1, max_iterations + 1):
                # Determine which prompt to use for this iteration
                if iteration_num == 1:
                    # First iteration uses original prompt
                    current_prompt = scenario["template"]
                    if scenario.get("json_output", False) and scenario.get(
                        "response_template"
                    ):
                        current_prompt = (
                            f"{current_prompt}\n\n{scenario['response_template']}"
                        )
                elif iteration_num <= len(iteration_rows):
                    # Use the prompt from the iteration process
                    # Note: This is a simplified approach - in reality we'd need to store
                    # intermediate prompts from the iterative improvement process
                    if iteration_num == max_iterations:
                        # Last iteration uses the best prompt
                        current_prompt = best_prompt_info["best_prompt"]
                    else:
                        # For intermediate iterations, we'll use the best prompt for now
                        # TODO: Store intermediate prompts from the iterative process
                        current_prompt = best_prompt_info["best_prompt"]
                else:
                    # Fallback to best prompt
                    current_prompt = best_prompt_info["best_prompt"]

                # Run this iteration's prompt multiple times
                for run in range(n_runs):
                    if verbose:
                        print(
                            f"\n--- Running iteration {iteration_num}, run {run + 1}/{n_runs} ---"
                        )

                    # Use the current iteration's prompt
                    raw_data_rows = process_scenario_with_final_prompt(
                        scenario=scenario,
                        data=data,
                        annotation_columns=annotation_columns,
                        labels=labels,
                        final_prompt=current_prompt,
                        verbose=verbose,
                    )

                    # Add run and iteration information to each row
                    for row in raw_data_rows:
                        row["run"] = run + 1
                        row["prompt_iteration"] = (
                            iteration_num  # Add prompt iteration info
                        )
                        row["prompt_name"] = scenario["prompt_name"]
                        row["use_validation_set"] = scenario.get(
                            "use_validation_set", True
                        )
                        row["best_accuracy"] = best_prompt_info["best_accuracy"]
                        row["total_iterations"] = len(iteration_rows)
                        row["is_best_iteration"] = (
                            iteration_num == max_iterations
                        )  # Mark if this is the best iteration
                        all_detailed_results.append(row)

    # Create DataFrame from the results
    detailed_results_df = pd.DataFrame(all_detailed_results)

    # Print the columns to debug
    if verbose:
        print("\nColumns in detailed_results_df:")
        print(detailed_results_df.columns.tolist())

    return detailed_results_df
