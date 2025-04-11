"""
Module for handling evaluation functionality in the Streamlit app.
"""

import streamlit as st
import pandas as pd
from typing import Any

from qualitative_analysis import compute_all_kappas, run_alt_test_general
from qualitative_analysis.evaluation import compute_classification_metrics


def compare_with_external_judgments(app_instance: Any) -> None:
    """
    Step 7: Compare with External Judgments (Annotation Columns)
    """
    st.markdown("### Step 7: Evaluate Model Performance", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 7", expanded=True):
        if not app_instance.results:
            st.warning("No analysis results. Please run the analysis first.")
            return

        if not app_instance.annotation_columns:
            st.info("No annotation columns were selected in Step 2.")
            return

        results_df: pd.DataFrame = st.session_state["results_df"]

        st.markdown(
            """
        This step provides three options to measure how closely your LLM's outputs align with existing
        manually annotated labels. If the alignment is sufficiently high, you could rely on the
        model-generated labels for annotating the rest of your unannotated data.

        We provide three options for comparing the LLM's outputs with human annotations: Cohen's Kappa,
        Classification Metrics, and the Alternative Annotator Test (Alt-Test).

        **Which Should I Choose?**
        - If you want to measure agreement between annotators:
            Cohen's Kappa is the simpler approach.
        - If you need detailed per-class performance metrics (recall, true positives, false positives):
            Classification Metrics provides a breakdown of model performance for each class.
        - If you have multiple annotation columns (≥ 3), want to see if the model
            "outperforms" or "can replace" humans, and can afford 50–100 annotated items:
            use the Alt-Test. This is more stringent because it compares against each
            annotator in a leave-one-out manner.

        In all cases, ideally 50+ annotated instances to get a stable estimate

        The ultimate decision of whether a metric is “good enough” depends on your
        research domain and practical considerations like cost, effort, and the
        consequences of annotation mistakes.

        If you are not satisfied with the model’s performance, you can go back to
        Step 3 and adjust the codebook and examples.

        Below, you can select your method, configure any needed parameters, and run
        the computation.
        """
        )

        # Choose the comparison method
        method: str = st.radio(
            "Select Comparison Method:",
            (
                "Cohen's Kappa (Agreement Analysis)",
                "Classification Metrics (Balanced Acc, TP%, FP%)",
                "Alt-Test (Model Viability)",
            ),
            index=0,
        )

        # --------------------------------------------------------------------------------
        # OPTION 1: COHEN'S KAPPA (AGREEMENT ANALYSIS)
        # --------------------------------------------------------------------------------
        if method == "Cohen's Kappa (Agreement Analysis)":
            st.markdown(
                """
                Analyze agreement between LLM and human annotators, as well as agreement among human annotators.
                
                This analysis provides:
                - Mean agreement between LLM and all human annotators (when multiple annotators are available)
                - Mean agreement among human annotators (when multiple annotators are available)
                - Individual agreement scores for all comparisons
                
                **Weighting Options:**
                - **Unweighted**: Treats all disagreements equally (e.g., disagreeing between 0 and 1 is the same as between 0 and 2)
                - **Linear**: Weights disagreements by their distance (e.g., disagreeing between 0 and 2 is twice as bad as between 0 and 1)
                - **Quadratic**: Weights disagreements by the square of their distance (e.g., disagreeing between 0 and 2 is four times as bad as between 0 and 1)
                
                Use weighting when your categories have a meaningful order (e.g., 0, 1, 2) and the magnitude of disagreement matters.
                """
            )

            # Add weighting option
            weights_option = st.radio(
                "Weighting Scheme:",
                ["Unweighted", "Linear", "Quadratic"],
                index=0,
                key="kappa_weights_option",
                help="Select how to weight disagreements between categories. Use weighting when categories have a meaningful order.",
            )

            # Convert the option to the format expected by the function
            weights_map = {
                "Unweighted": None,
                "Linear": "linear",
                "Quadratic": "quadratic",
            }
            weights = weights_map[weights_option]

            # LLM columns presumably the ones in selected_fields
            llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]

            if not llm_columns:
                st.warning("No LLM-generated columns found in the results to compare.")
                return

            if len(app_instance.annotation_columns) < 1:
                st.warning(
                    "At least one annotation column is required for agreement analysis."
                )
                return

            # Figure out a default index that points to our label column if possible
            default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in llm_columns:
                    default_index = llm_columns.index(label_col)

            llm_judgment_col: str = st.selectbox(
                "Select LLM Judgment Column:",
                llm_columns,
                index=default_index,
                key="llm_judgment_col_select",
            )

            if st.button("Compute Agreement Scores", key="compute_agreement_button"):
                if llm_judgment_col not in results_df.columns:
                    st.error("The chosen LLM column is not in the results dataframe.")
                    return

                # Check if annotation columns exist in the dataframe
                missing_columns = [
                    col
                    for col in app_instance.annotation_columns
                    if col not in results_df.columns
                ]
                if missing_columns:
                    st.error(
                        f"The following annotation columns are not in the results dataframe: {', '.join(missing_columns)}"
                    )
                    return

                # Prepare data for analysis
                analysis_data = results_df[
                    [llm_judgment_col] + app_instance.annotation_columns
                ].copy()

                # Convert to integers if possible
                try:
                    for col in [llm_judgment_col] + app_instance.annotation_columns:
                        analysis_data[col] = analysis_data[col].astype(int)
                except ValueError:
                    st.error(
                        "Could not convert columns to integer for agreement analysis."
                    )
                    return

                # Drop rows with missing values
                analysis_data = analysis_data.dropna()

                if analysis_data.empty:
                    st.error(
                        "No valid (non-NA) rows found after filtering for these columns."
                    )
                    return

                # Create a dictionary of human annotations for compute_all_kappas
                human_annotations = {}
                for col in app_instance.annotation_columns:
                    human_annotations[col] = analysis_data[col].tolist()

                # Compute all kappa scores with the selected weighting
                kappa_scores = compute_all_kappas(
                    analysis_data[llm_judgment_col].tolist(),
                    human_annotations,
                    weights=weights,
                )

                # Calculate mean LLM-Human agreement
                llm_human_scores = [
                    score
                    for key, score in kappa_scores.items()
                    if key.startswith("model_vs_")
                ]
                mean_llm_human = (
                    sum(llm_human_scores) / len(llm_human_scores)
                    if llm_human_scores
                    else 0
                )

                # Calculate mean Human-Human agreement (only if we have multiple human annotators)
                human_human_scores = [
                    score
                    for key, score in kappa_scores.items()
                    if not key.startswith("model_vs_")
                ]
                mean_human_human = (
                    sum(human_human_scores) / len(human_human_scores)
                    if human_human_scores
                    else 0
                )

                # Display appropriate scores based on number of annotators
                if len(app_instance.annotation_columns) > 1:
                    # Multiple annotators - show mean scores
                    st.subheader("Mean Agreement Scores")
                    st.write(f"**Mean LLM-Human Agreement**: {mean_llm_human:.4f}")
                    st.write(f"**Mean Human-Human Agreement**: {mean_human_human:.4f}")
                else:
                    # Single annotator - show individual score
                    st.subheader("Agreement Score")
                    human_annotator = app_instance.annotation_columns[0]
                    kappa_score = next(
                        (
                            score
                            for key, score in kappa_scores.items()
                            if key == f"model_vs_{human_annotator}"
                        ),
                        0,
                    )
                    st.write(
                        f"**Cohen's Kappa (LLM vs {human_annotator})**: {kappa_score:.4f}"
                    )

                # Display individual scores in a table (only if we have multiple annotators)
                if len(app_instance.annotation_columns) > 1:
                    st.subheader("Individual Agreement Scores")

                    # Create dataframe for LLM-Human comparisons
                    llm_human_data = []
                    for key, score in kappa_scores.items():
                        if key.startswith("model_vs_"):
                            human_annotator = key.replace("model_vs_", "")
                            llm_human_data.append(
                                {
                                    "Human Annotator": human_annotator,
                                    "Cohen's Kappa": f"{score:.4f}",
                                }
                            )

                    if llm_human_data:
                        st.write("**LLM vs Human Annotators**")
                        st.table(pd.DataFrame(llm_human_data))

                    # Create dataframe for Human-Human comparisons
                    human_human_data = []
                    for key, score in kappa_scores.items():
                        if not key.startswith("model_vs_"):
                            annotators = key.split("_vs_")
                            human_human_data.append(
                                {
                                    "Annotator 1": annotators[0],
                                    "Annotator 2": annotators[1],
                                    "Cohen's Kappa": f"{score:.4f}",
                                }
                            )

                    if human_human_data:
                        st.write("**Human vs Human Annotators**")
                        st.table(pd.DataFrame(human_human_data))

        # --------------------------------------------------------------------------------
        # OPTION 2: CLASSIFICATION METRICS
        # --------------------------------------------------------------------------------
        elif method == "Classification Metrics (Balanced Acc, TP%, FP%)":
            st.markdown(
                """
                Analyze detailed classification metrics for each class, focusing on recall and confusion matrix elements.
                
                This analysis uses majority vote from human annotations as ground truth and provides:
                - Class distribution (number of instances per class)
                - Global metrics for the LLM and each human annotator (balanced accuracy and F1 score)
                - Per-class metrics showing:
                  - TP%: Percentage of instances of this class that were correctly identified
                  - FP%: Percentage of predictions for this class that were incorrect
                """
            )

            # LLM columns presumably the ones in selected_fields
            metrics_llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]

            if not metrics_llm_columns:
                st.warning("No LLM-generated columns found in the results to compare.")
                return

            if len(app_instance.annotation_columns) < 1:
                st.warning(
                    "At least one annotation column is required for classification metrics."
                )
                return

            # Figure out a default index that points to our label column if possible
            metrics_default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in metrics_llm_columns:
                    metrics_default_index = metrics_llm_columns.index(label_col)

            metrics_llm_judgment_col: str = st.selectbox(
                "Select LLM Judgment Column:",
                metrics_llm_columns,
                index=metrics_default_index,
                key="llm_metrics_col_select",
            )

            if st.button(
                "Compute Classification Metrics", key="compute_metrics_button"
            ):
                if metrics_llm_judgment_col not in results_df.columns:
                    st.error("The chosen LLM column is not in the results dataframe.")
                    return

                # Check if annotation columns exist in the dataframe
                missing_columns = [
                    col
                    for col in app_instance.annotation_columns
                    if col not in results_df.columns
                ]
                if missing_columns:
                    st.error(
                        f"The following annotation columns are not in the results dataframe: {', '.join(missing_columns)}"
                    )
                    return

                # Prepare data for analysis
                analysis_data = results_df[
                    [metrics_llm_judgment_col] + app_instance.annotation_columns
                ].copy()

                # Convert to integers if possible
                try:
                    for col in [
                        metrics_llm_judgment_col
                    ] + app_instance.annotation_columns:
                        analysis_data[col] = analysis_data[col].astype(int)
                except ValueError:
                    st.error(
                        "Could not convert columns to integer for classification metrics."
                    )
                    return

                # Drop rows with missing values
                analysis_data = analysis_data.dropna()

                if analysis_data.empty:
                    st.error(
                        "No valid (non-NA) rows found after filtering for these columns."
                    )
                    return

                # Create a dictionary of human annotations
                human_annotations = {}
                for col in app_instance.annotation_columns:
                    human_annotations[col] = analysis_data[col].tolist()

                # Compute classification metrics
                metrics_results = compute_classification_metrics(
                    analysis_data[metrics_llm_judgment_col].tolist(), human_annotations
                )

                # Display class distribution
                st.subheader("Class Distribution")
                class_dist = metrics_results["class_distribution"]

                # Create a dataframe for class distribution
                class_dist_data = []
                for label, count in class_dist.items():
                    class_dist_data.append(
                        {
                            "Class": label,
                            "Count": count,
                            "Percentage": f"{(count / sum(class_dist.values())) * 100:.1f}%",
                        }
                    )

                st.table(pd.DataFrame(class_dist_data))

                # Display global metrics
                st.subheader("Global Metrics")

                # Create dataframe for global metrics with balanced accuracy and F1 score
                global_metrics_data = []
                for rater, metrics in metrics_results["global_metrics"].items():
                    # Calculate balanced accuracy as the average recall across all classes
                    class_recalls = []
                    class_precisions = []
                    class_f1s = []

                    for label in metrics_results["per_class_metrics"].keys():
                        if rater in metrics_results["per_class_metrics"][label]:
                            # Get recall (TP / (TP + FN))
                            recall = metrics_results["per_class_metrics"][label][rater][
                                "recall"
                            ]
                            class_recalls.append(recall)

                            # Calculate precision (TP / (TP + FP))
                            tp = metrics_results["per_class_metrics"][label][rater][
                                "correct_count"
                            ]
                            fp = metrics_results["per_class_metrics"][label][rater][
                                "false_positives"
                            ]
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            class_precisions.append(precision)

                            # Calculate F1 score for this class
                            f1 = (
                                2 * (precision * recall) / (precision + recall)
                                if (precision + recall) > 0
                                else 0
                            )
                            class_f1s.append(f1)

                    # Calculate balanced accuracy and macro F1 score
                    balanced_acc = (
                        sum(class_recalls) / len(class_recalls) if class_recalls else 0
                    )
                    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0

                    global_metrics_data.append(
                        {
                            "Annotator": "LLM" if rater == "model" else rater,
                            "Balanced Accuracy": f"{balanced_acc:.4f}",
                            "F1 Score": f"{macro_f1:.4f}",
                        }
                    )

                st.table(pd.DataFrame(global_metrics_data))

                # Display per-class metrics
                st.subheader("Per-Class Metrics")

                # For each class, show metrics for LLM and humans
                for label in sorted(metrics_results["per_class_metrics"].keys()):
                    st.write(f"**Class {label}** (Count: {class_dist.get(label, 0)})")

                    # Create dataframe for this class's metrics with percentages for TP and FP
                    class_metrics_data = []
                    for rater, metrics_obj in metrics_results["per_class_metrics"][
                        label
                    ].items():
                        # Calculate TP and FP percentages
                        tp_count = metrics_obj["correct_count"]  # type: ignore
                        fp_count = metrics_obj["false_positives"]  # type: ignore

                        # TP% is the same as recall
                        tp_percentage = metrics_obj["recall"]  # type: ignore

                        # FP% = FP / (TP + FP) - percentage of predictions that were false positives
                        total_predictions = tp_count + fp_count
                        fp_percentage = (
                            fp_count / total_predictions if total_predictions > 0 else 0
                        )

                        class_metrics_data.append(
                            {
                                "Annotator": "LLM" if rater == "model" else rater,
                                "TP%": f"{tp_percentage:.2%}",
                                "FP%": f"{fp_percentage:.2%}",
                            }
                        )

                    st.table(pd.DataFrame(class_metrics_data))

        # --------------------------------------------------------------------------------
        # OPTION 3: ALT-TEST
        # --------------------------------------------------------------------------------
        else:  # method == "Alt-Test (Model Viability)"
            st.markdown(
                """
                **Alternative Annotator Test** (requires >= 3 annotation columns).  
                Compares the model's predictions to human annotators by excluding one human at a time 
                and measuring alignment with the remaining humans.
                
                - The test yields a "winning rate" (the proportion of annotators for
                   which the LLM outperforms or is at least as good as that annotator,
                   given the cost/benefit trade-off).
                 - "Epsilon" (ε), represents how much we adjust the human advantage to account for time/cost/effort
                   savings when using an LLM. Larger ε values make it easier for the LLM
                   to "pass" because it reflects that human annotations are costlier (if your human are experts, the original article recommend 0.2, if they are crowdworker, 0.1).
                 - If the LLM's winning rate ≥ 0.5, the test concludes that the LLM is
                   (statistically) as viable as a human annotator for that dataset (the LLM is "better" than half the humans).
                """
            )

            if len(app_instance.annotation_columns) < 3:
                st.warning(
                    "You must have at least 3 annotation columns to run the alt-test."
                )
                return

            alt_llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]
            if not alt_llm_columns:
                st.warning("No valid LLM columns found in the results.")
                return

            # Default to the label column if possible
            alt_default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in alt_llm_columns:
                    alt_default_index = alt_llm_columns.index(label_col)

            alt_model_col: str = st.selectbox(
                "Choose model column for alt-test:",
                alt_llm_columns,
                index=alt_default_index,
                key="alt_test_model_col_select",
            )

            metric_choice: str = st.selectbox(
                "Alignment Metric:",
                ["accuracy", "rmse"],
                index=0,
                key="alt_test_metric_choice",
            )

            epsilon_val: float = st.number_input(
                "Epsilon (cost-benefit margin)",
                min_value=0.0,
                value=0.1,
                step=0.01,
                key="alt_test_epsilon",
            )
            alpha_val: float = st.number_input(
                "Alpha (significance level)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                key="alt_test_alpha",
            )

            if st.button("Run Alternative Annotator Test", key="run_alt_test_button"):
                try:
                    alt_results: dict[str, Any] = run_alt_test_general(
                        df=results_df,
                        annotation_columns=app_instance.annotation_columns,
                        model_col=alt_model_col,
                        epsilon=epsilon_val,
                        alpha=alpha_val,
                        metric=metric_choice,
                        verbose=False,  # We'll display below
                    )
                except ValueError as ve:
                    st.error(f"Alt-Test Error: {ve}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    return

                # Display results
                st.subheader("Alt-Test Results")

                ann_cols: list[str] = alt_results["annotator_columns"]
                pvals: list[float] = alt_results["pvals"]
                rejections: list[bool] = alt_results["rejections"]
                rho_f: list[float] = alt_results["rho_f"]
                rho_h: list[float] = alt_results["rho_h"]

                table_data: list[dict[str, Any]] = []
                for ann, pv, rj, rf, rh in zip(
                    ann_cols, pvals, rejections, rho_f, rho_h
                ):
                    row = {
                        "Annotator": ann,
                        "p-value": f"{pv:.4f}" if pd.notna(pv) else "NaN",
                        "RejectH0?": rj,
                        "rho_f (LLM advantage)": f"{rf:.3f}" if pd.notna(rf) else "NaN",
                        "rho_h (Human advantage)": (
                            f"{rh:.3f}" if pd.notna(rh) else "NaN"
                        ),
                    }
                    table_data.append(row)

                st.write(pd.DataFrame(table_data))

                st.write(
                    f"**Winning Rate (omega):** {alt_results['winning_rate']:.3f}  "
                    f"**Average LLM Advantage (rho):** {alt_results['average_advantage_probability']:.3f}"
                )

                if alt_results["passed_alt_test"]:
                    st.success(
                        "✅ The model **passed** the alt-test (winning rate ≥ 0.5)."
                    )
                else:
                    st.warning(
                        "❌ The model **did not pass** the alt-test (winning rate < 0.5)."
                    )
