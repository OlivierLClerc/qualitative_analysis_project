"""
Module for handling evaluation functionality in the Streamlit app.
"""

import streamlit as st
import pandas as pd
from typing import Any

from qualitative_analysis import compute_cohens_kappa, run_alt_test_general


def compare_with_external_judgments(app_instance: Any) -> None:
    """
    Step 7: Compare with External Judgments (Annotation Columns)

    This step provides two options to measure how closely your LLM’s outputs align with existing
    manually annotated labels. If the alignment is sufficiently high, you could rely on the
    model-generated labels for annotating the rest of your unannotated data.

    We provide two options for comparing the LLM's outputs with human annotations, Cohen's Kappa
    and the Alternative Annotator Test (Alt-Test).

    **Which Should I Choose?**
      - If you have only one set of human labels or want a quick agreement measure:
        Cohen’s Kappa is the simpler approach.
      - If you have multiple annotation columns (≥ 3), want to see if the model
        “outperforms” or “can replace” humans, and can afford 50–100 annotated items:
        use the Alt-Test. This is more stringent because it compares against each
        annotator in a leave-one-out manner.

    In both cases, ideally 50+ annotated instances to get a stable estimate

    The ultimate decision of whether a metric is “good enough” depends on your
    research domain and practical considerations like cost, effort, and the
    consequences of annotation mistakes.

    If you are not satisfied with the model’s performance, you can go back to
    Step 3 and adjust the codebook and examples.

    Below, you can select your method, configure any needed parameters, and run
    the computation.
    """
    st.header("Step 7: Compare with Annotation Columns")

    if not app_instance.results:
        st.warning("No analysis results. Please run the analysis first.")
        return

    if not app_instance.annotation_columns:
        st.info("No annotation columns were selected in Step 2.")
        return

    results_df: pd.DataFrame = st.session_state["results_df"]

    # Choose the comparison method
    method: str = st.radio(
        "Select Comparison Method:",
        ("Cohen's Kappa", "Alt-Test"),
        index=0,
    )

    # --------------------------------------------------------------------------------
    # OPTION 1: COHEN'S KAPPA
    # --------------------------------------------------------------------------------
    if method == "Cohen's Kappa":
        st.markdown(
            """
            Compare a single LLM-generated column with a single human annotation column (numeric).
            Will compute **Cohen's Kappa**.
            """
        )

        # LLM columns presumably the ones in selected_fields
        llm_columns: list[str] = [
            col for col in app_instance.selected_fields if col in results_df.columns
        ]

        if not llm_columns:
            st.warning("No LLM-generated columns found in the results to compare.")
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

        annotation_col: str = st.selectbox(
            "Select Annotation Column:",
            app_instance.annotation_columns,
            key="annotation_col_select",
        )

        if st.button("Compute Cohen's Kappa", key="compute_kappa_button"):
            if llm_judgment_col not in results_df.columns:
                st.error("The chosen LLM column is not in the results dataframe.")
                return

            if annotation_col not in results_df.columns:
                st.error(
                    "The chosen annotation column is not in the results dataframe."
                )
                return

            merged_aligned: pd.DataFrame = results_df[
                [llm_judgment_col, annotation_col]
            ].dropna()

            try:
                # Attempt integer conversion
                merged_aligned[llm_judgment_col] = merged_aligned[
                    llm_judgment_col
                ].astype(int)
                merged_aligned[annotation_col] = merged_aligned[annotation_col].astype(
                    int
                )
            except ValueError:
                st.error("Could not convert columns to integer for Cohen's Kappa.")
                return

            if merged_aligned.empty:
                st.error(
                    "No valid (non-NA) rows found after filtering for these columns."
                )
                return

            kappa: float = compute_cohens_kappa(
                merged_aligned[llm_judgment_col],
                merged_aligned[annotation_col],
            )
            st.write(f"**Cohen's Kappa**: {kappa:.4f}")

    # --------------------------------------------------------------------------------
    # OPTION 2: ALT-TEST
    # --------------------------------------------------------------------------------
    else:  # method == "Alt-Test"
        st.markdown(
            """
            **Alternative Annotator Test** (requires >= 3 annotation columns).  
            Compares the model's predictions to human annotators by excluding one human at a time 
            and measuring alignment with the remaining humans.
            
            - The test yields a “winning rate” (the proportion of annotators for
               which the LLM outperforms or is at least as good as that annotator,
               given the cost/benefit trade-off).
             - “Epsilon” (ε), represents how much we adjust the human advantage to account for time/cost/effort
               savings when using an LLM. Larger ε values make it easier for the LLM
               to “pass” because it reflects that human annotations are costlier (if your human are experts, the original article recommend 0.2, if they are crowdworker, 0.1).
             - If the LLM’s winning rate ≥ 0.5, the test concludes that the LLM is
               (statistically) as viable as a human annotator for that dataset (the LLM is "better" than half the humans).
            """
        )

        if len(app_instance.annotation_columns) < 3:
            st.warning(
                "You must have at least 3 annotation columns to run the alt-test."
            )
            return

        alt_test_llm_columns: list[str] = [
            col for col in app_instance.selected_fields if col in results_df.columns
        ]
        if not alt_test_llm_columns:
            st.warning("No valid LLM columns found in the results.")
            return

        # Default to the label column if possible
        default_model_index = 0
        if "label_column" in st.session_state:
            label_col = st.session_state["label_column"]
            if label_col in alt_test_llm_columns:
                default_model_index = alt_test_llm_columns.index(label_col)

        model_col: str = st.selectbox(
            "Choose model column for alt-test:",
            alt_test_llm_columns,
            index=default_model_index,
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
                    model_col=model_col,
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
            for ann, pv, rj, rf, rh in zip(ann_cols, pvals, rejections, rho_f, rho_h):
                row = {
                    "Annotator": ann,
                    "p-value": f"{pv:.4f}" if pd.notna(pv) else "NaN",
                    "RejectH0?": rj,
                    "rho_f (LLM advantage)": f"{rf:.3f}" if pd.notna(rf) else "NaN",
                    "rho_h (Human advantage)": f"{rh:.3f}" if pd.notna(rh) else "NaN",
                }
                table_data.append(row)

            st.write(pd.DataFrame(table_data))

            st.write(
                f"**Winning Rate (omega):** {alt_results['winning_rate']:.3f}  "
                f"**Average LLM Advantage (rho):** {alt_results['average_advantage_probability']:.3f}"
            )

            if alt_results["passed_alt_test"]:
                st.success("✅ The model **passed** the alt-test (winning rate ≥ 0.5).")
            else:
                st.warning(
                    "❌ The model **did not pass** the alt-test (winning rate < 0.5)."
                )
