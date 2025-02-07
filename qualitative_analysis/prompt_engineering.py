"""
prompt_engineering.py

This module provides helper functions for prompt engineering and the iterative improvement of prompts
used in classification tasks with large language models (LLMs). It includes utilities for:
  - Detecting discrepancies between LLM predictions and human labels.
  - Extracting examples where the LLM's predictions agree with human labels.
  - Calling a secondary LLM (LLM2) to suggest improvements to an existing prompt.
  - Running an iterative loop to refine a prompt based on training and validation data performance.
  - Maintaining a history of previous prompts with their validation performance and the changes made.

Dependencies:
    - json
    - pandas
    - time
    - Any external functions:
         * get_llm_client
         * process_general_verbatims
         * compute_cohens_kappa
         * accuracy_score
         * run_alt_test_general
    - config (for MODEL_CONFIG)
    - typing: List, Dict, Any, Optional, Tuple
"""

import json
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import List, Dict, Any, Optional, Tuple
from qualitative_analysis.model_interaction import get_llm_client
from qualitative_analysis.alt_test import run_alt_test_general
from qualitative_analysis.notebooks_functions import process_general_verbatims
from qualitative_analysis.evaluation import compute_cohens_kappa
import qualitative_analysis.config as config


def find_discrepancies(df: pd.DataFrame, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Identifies discrepancies between the model predictions and human-provided labels within a DataFrame.

    This function iterates over each row in the DataFrame and compares the values in the columns
    "ModelPrediction" and "div_rater1". For any row where these two values differ, it records a dictionary
    capturing the associated verbatim text, the human label, and the LLM's prediction.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing at least the following columns:
          - "verbatim": The original text input.
          - "div_rater1": The human (gold-standard) label.
          - "ModelPrediction": The label predicted by the model.
    verbose : bool, optional
        If True, prints the number of discrepancies found. Default is True.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries. Each dictionary has the keys:
            - "verbatim": The input text.
            - "human_label": The human-provided label.
            - "llm1_label": The label predicted by the model.
    """
    discrepancies = []
    for _, row in df.iterrows():
        if row["ModelPrediction"] != row["div_rater1"]:
            discrepancies.append(
                {
                    "verbatim": row["verbatim"],
                    "human_label": row["div_rater1"],
                    "llm1_label": row["ModelPrediction"],
                }
            )
    if verbose:
        print(f"Found {len(discrepancies)} discrepancies.")
    return discrepancies


def find_similarities(df: pd.DataFrame, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Extracts examples where the model's prediction matches the human-provided label.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing at least the following columns:
          - "verbatim": The original text input.
          - "div_rater1": The human (gold-standard) label.
          - "ModelPrediction": The label predicted by the model.
    verbose : bool, optional
        If True, prints the number of similarities found. Default is True.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries for examples where the model prediction is correct.
    """
    similarities = []
    for _, row in df.iterrows():
        if row["ModelPrediction"] == row["div_rater1"]:
            similarities.append(
                {
                    "verbatim": row["verbatim"],
                    "human_label": row["div_rater1"],
                    "llm1_label": row["ModelPrediction"],
                }
            )
    if verbose:
        print(f"Found {len(similarities)} similarities.")
    return similarities


def call_llm2_for_improvement(
    llm2_client,
    llm2_model_name: str,
    current_prompt: str,
    example_set: Dict[str, List[Dict[str, Any]]],
    prompt_history: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    verbose: bool = True,
    json_output: bool = False,
    response_template: str = "",
) -> Optional[Dict[str, str]]:
    """
    Calls a secondary LLM (LLM2) to generate an improved prompt along with a summary of changes.

    If json_output is True and a response_template is provided, the function first removes the
    response_template from current_prompt (so that it isn’t iteratively modified) and then later
    re-appends it to the new prompt.

    Returns a dictionary with keys "new_prompt" and "changes" if successful.
    """
    history_text = json.dumps(prompt_history, indent=2) if prompt_history else "None"

    # Remove the response template from current_prompt if needed
    if json_output and response_template:
        base_prompt = current_prompt.replace(response_template, "").strip()
    else:
        base_prompt = current_prompt

    instructions = f"""
You are an assistant tasked with improving a codebook used for classification.
You are provided with:
  - The current prompt.
  - A set of examples divided into:
      - "bad_examples": Instances where LLM1's prediction did not match the human label.
      - "good_examples": Instances where LLM1's prediction matched the human label.
  - A history of previous iterations with their validation performance and the changes made:
{history_text}

Your task is to analyze the classification task, the provided examples, and the history of previous prompts.
Based on this analysis, produce a new prompt that would help LLM1 improve its predictions.
Also, provide a summary of the major changes you made compared to the previous prompt and explain their expected impact on validation accuracy.
Make sure that in your revised prompt you preserve exactly the placeholder line: Input:
{{verbatim_text}}

Here is the current prompt:
{base_prompt}

Here is the set of examples:
{json.dumps(example_set, indent=2)}

Please output a JSON object in the following format exactly:

{{
  "new_prompt": "Your revised prompt here.",
  "changes": "A description of the major changes you made and their expected impact on validation accuracy."
}}

IMPORTANT:
- Do not include backticks or triple quotes.
- Do not include any additional keys or text outside the JSON object.
- The JSON must be valid and parseable by Python's json.loads.
- You MUST preserve exactly the placeholder line: Input:
{{verbatim_text}} in your revised prompt.
"""
    llm2_response_text, _ = llm2_client.get_response(
        prompt=instructions,
        model=llm2_model_name,
        max_tokens=10000,
        temperature=temperature,
        verbose=verbose,
    )
    if verbose:
        print("LLM2 raw response:\n", llm2_response_text)
    try:
        data = json.loads(llm2_response_text)
        revised_prompt = data.get("new_prompt", "").strip()
        changes = data.get("changes", "").strip()
        if revised_prompt and changes:
            if verbose:
                print(
                    "Revised prompt and changes successfully extracted from LLM2 JSON."
                )
            # Re-append the response_template so that the response format stays constant.
            if json_output and response_template:
                revised_prompt = f"{revised_prompt}\n\n{response_template}"
            return {"new_prompt": revised_prompt, "changes": changes}
        else:
            if verbose:
                print("JSON parsing succeeded, but one or both keys are empty.")
            return None
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON from LLM2 response: {e}")
        return None


def run_iterative_prompt_improvement(
    scenario: Dict[str, Any],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    alt_test: bool = True,
    errors_examples: float = 0.6,  # Fraction of examples that should be error examples
    examples_to_give: int = 10,  # Maximum total number of examples to pass to LLM2
    json_output: bool = False,
    selected_fields: Optional[List[str]] = None,
    response_template: str = "",
    verbose: bool = True,
) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Iteratively refines a prompt based on discrepancies observed between model predictions and human labels,
    using two LLMs. When json_output is True, the constant response_template is appended to the base prompt
    when sending it to LLM1. Likewise, before sending the prompt to LLM2 for improvements the response_template
    is removed, and then re-appended to keep the response format consistent.

    The selected_fields parameter is used (in your downstream processing) to extract the desired keys
    from LLM1's JSON response.

    If annotation_columns is provided (e.g. ['div_rater1', 'div_rater2', 'div_rater3']), the ground truth
    will be computed as the majority vote among these columns.

    The labels parameter specifies the allowed label set and is used when computing Cohen's kappa.

    Returns:
      - best_prompt: The best prompt found.
      - best_accuracy: The best validation accuracy achieved.
      - iteration_rows: A list of dictionaries capturing metrics from each iteration.
    """
    provider_1 = scenario["provider_llm1"]
    model_name_1 = scenario["model_name_llm1"]
    temperature_llm1 = scenario["temperature_llm1"]
    prefix_llm1 = scenario.get("prefix", None)

    provider_2 = scenario["provider_llm2"]
    model_name_2 = scenario["model_name_llm2"]
    temperature_llm2 = scenario["temperature_llm2"]

    prompt_name = scenario.get("prompt_name", "default_prompt")
    max_iterations = scenario.get("max_iterations", 3)
    n_completions = scenario.get("n_completions", 1)
    initial_prompt = scenario["template"]

    scenario_info = {
        "data_set": scenario.get("data_set", "default_data_set"),
        "N_train": len(train_data),
        "N_val": len(val_data),
        "provider": provider_1,
        "model_name": model_name_1,
        "temperature": temperature_llm1,
        "prompt_name": prompt_name,
    }

    # Compute majority vote as ground truth using the provided annotation columns.
    if annotation_columns and len(annotation_columns) > 0:
        train_data["GroundTruth"] = train_data.apply(
            lambda row: row[annotation_columns].value_counts().idxmax(), axis=1
        )
        val_data["GroundTruth"] = val_data.apply(
            lambda row: row[annotation_columns].value_counts().idxmax(), axis=1
        )
        ground_truth_column = "GroundTruth"
    else:
        raise ValueError("You must provide annotation columns for majority voting.")

    # Initialize LLM clients.
    llm1_client = get_llm_client(
        provider=provider_1, config=config.MODEL_CONFIG[provider_1]
    )
    llm2_client = get_llm_client(
        provider=provider_2, config=config.MODEL_CONFIG[provider_2]
    )

    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_accuracy = -1.0
    iteration_rows = []
    prompt_history: List[Dict[str, Any]] = []

    prev_accuracy_val: Optional[float] = None
    last_changes: Optional[str] = None

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n=== Iteration {iteration}/{max_iterations} ===")

        if json_output and response_template:
            full_prompt = f"{current_prompt}\n\n{response_template}"
        else:
            full_prompt = current_prompt

        # Evaluate on training set.
        start_time = time.time()
        train_pred_df, train_cost_info, train_totals = process_general_verbatims(
            verbatims_subset=train_data["verbatim"].tolist(),
            llm_client=llm1_client,
            model_name=model_name_1,
            prompt_template=full_prompt,
            prefix=prefix_llm1,
            temperature=temperature_llm1,
            verbose=verbose,
            json_output=json_output,
            selected_fields=selected_fields,
            n_completions=n_completions,
        )
        end_time = time.time()
        train_tokens = train_totals["total_tokens_used"]
        train_cost = train_totals["total_cost"]
        train_time_s = end_time - start_time

        train_data["ModelPrediction"] = train_pred_df["Label"].values
        y_true_train = train_data[ground_truth_column].tolist()
        y_pred_train = train_data["ModelPrediction"].fillna(-1).tolist()
        accuracy_train = accuracy_score(y_true_train, y_pred_train)
        kappa_train = compute_cohens_kappa(y_true_train, y_pred_train, labels=labels)

        if alt_test:
            alt_test_res_train = run_alt_test_general(
                df=train_data,
                annotation_columns=annotation_columns,
                model_col="ModelPrediction",
                epsilon=0.1,
                alpha=0.05,
                verbose=verbose,
            )
            winning_rate_train = alt_test_res_train["winning_rate"]
            passed_alt_test_train = alt_test_res_train["passed_alt_test"]
            avg_adv_prob_train = alt_test_res_train["average_advantage_probability"]
        else:
            winning_rate_train = passed_alt_test_train = avg_adv_prob_train = None

        # Evaluate on validation set.
        start_time_val = time.time()
        val_pred_df, val_cost_info, val_totals = process_general_verbatims(
            verbatims_subset=val_data["verbatim"].tolist(),
            llm_client=llm1_client,
            model_name=model_name_1,
            prompt_template=full_prompt,
            prefix=prefix_llm1,
            temperature=temperature_llm1,
            verbose=False,
            json_output=json_output,
            selected_fields=selected_fields,
            n_completions=n_completions,
        )
        end_time_val = time.time()
        val_tokens = val_totals["total_tokens_used"]
        val_cost = val_totals["total_cost"]
        val_time_s = end_time_val - start_time_val

        val_data["ModelPrediction"] = val_pred_df["Label"].values
        y_true_val = val_data[ground_truth_column].tolist()
        y_pred_val = val_data["ModelPrediction"].fillna(-1).tolist()
        accuracy_val = accuracy_score(y_true_val, y_pred_val)
        kappa_val = compute_cohens_kappa(y_true_val, y_pred_val, labels=labels)

        if alt_test:
            alt_test_res_val = run_alt_test_general(
                df=val_data,
                annotation_columns=annotation_columns,
                model_col="ModelPrediction",
                epsilon=0.1,
                alpha=0.05,
                verbose=verbose,
            )
            winning_rate_val = alt_test_res_val["winning_rate"]
            passed_alt_test_val = alt_test_res_val["passed_alt_test"]
            avg_adv_prob_val = alt_test_res_val["average_advantage_probability"]
        else:
            winning_rate_val = passed_alt_test_val = avg_adv_prob_val = None

        total_tokens = train_tokens + val_tokens
        total_cost = train_cost + val_cost

        if iteration == 1:
            effect = "N/A"
            changes_record = "Initial prompt"
        else:
            if prev_accuracy_val is not None:
                if accuracy_val > prev_accuracy_val:
                    effect = "Those changes improved the classification"
                elif accuracy_val < prev_accuracy_val:
                    effect = "Those changes decreased the classification"
                else:
                    effect = "No change in classification performance"
            else:
                effect = "N/A"
            changes_record = (
                last_changes if last_changes is not None else "No changes recorded"
            )

        history_entry = {
            "iteration": iteration,
            "accuracy_val": accuracy_val,
            "changes": changes_record,
            "effect_of_changes": effect,
        }
        prompt_history.append(history_entry)

        row = {
            "data_set": scenario_info["data_set"],
            "N_train": scenario_info["N_train"],
            "N_val": scenario_info["N_val"],
            "model": model_name_1,
            "prompt_name": prompt_name,
            "iteration": iteration,
            "kappa_train": kappa_train,
            "kappa_val": kappa_val,
            "accuracy_train": accuracy_train,
            "accuracy_val": accuracy_val,
            "winning_rate_train": winning_rate_train,
            "passed_alt_test_train": passed_alt_test_train,
            "avg_adv_prob_train": avg_adv_prob_train,
            "winning_rate_val": winning_rate_val,
            "passed_alt_test_val": passed_alt_test_val,
            "avg_adv_prob_val": avg_adv_prob_val,
            "tokens_used": total_tokens,
            "cost": total_cost,
            "running_time_s": train_time_s + val_time_s,
        }
        iteration_rows.append(row)

        if accuracy_val > best_accuracy:
            best_accuracy = accuracy_val
            best_prompt = current_prompt

        bad_examples = []
        good_examples = []
        for _, row_data in train_data.iterrows():
            if row_data["ModelPrediction"] != row_data[ground_truth_column]:
                bad_examples.append(
                    {
                        "verbatim": row_data["verbatim"],
                        "human_label": row_data[ground_truth_column],
                        "llm1_label": row_data["ModelPrediction"],
                    }
                )
            else:
                good_examples.append(
                    {
                        "verbatim": row_data["verbatim"],
                        "human_label": row_data[ground_truth_column],
                        "llm1_label": row_data["ModelPrediction"],
                    }
                )
        if verbose:
            print(
                f"Found {len(bad_examples)} discrepancies and {len(good_examples)} correct examples."
            )

        num_total = examples_to_give
        num_bad_desired = int(round(num_total * errors_examples))
        num_good_desired = num_total - num_bad_desired
        selected_bad = bad_examples[: min(len(bad_examples), num_bad_desired)]
        selected_good = good_examples[: min(len(good_examples), num_good_desired)]
        example_set = {"bad_examples": selected_bad, "good_examples": selected_good}

        if not bad_examples:
            if verbose:
                print("No discrepancies found; stopping iterative improvement.")
            break

        llm2_result = call_llm2_for_improvement(
            llm2_client=llm2_client,
            llm2_model_name=model_name_2,
            current_prompt=current_prompt,
            example_set=example_set,
            prompt_history=prompt_history,
            temperature=temperature_llm2,
            verbose=verbose,
            json_output=json_output,
            response_template=response_template,
        )
        if llm2_result:
            new_prompt = llm2_result["new_prompt"]
            changes = llm2_result["changes"]
            if verbose:
                print("LLM2 suggested changes:", changes)
            last_changes = changes
            if json_output and response_template:
                current_prompt = new_prompt.replace(response_template, "").strip()
            else:
                current_prompt = new_prompt
        else:
            if verbose:
                print("No new prompt returned; terminating iteration.")
            break

        prev_accuracy_val = accuracy_val

    if json_output and response_template:
        best_prompt = f"{best_prompt}\n\n{response_template}"
    return best_prompt, best_accuracy, iteration_rows
