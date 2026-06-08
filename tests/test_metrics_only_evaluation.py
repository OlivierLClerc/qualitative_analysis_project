import unittest
from unittest.mock import patch

import pandas as pd

import streamlit_app.evaluation_dashboard as evaluation_dashboard
from streamlit_app.evaluation_dashboard import (
    compute_classification_results,
    compute_classical_krippendorff_results,
    compute_kappa_results,
)
from streamlit_app.evaluation_mappings import EvaluationMapping
from streamlit_app.metrics_only_page import (
    autodetect_scale_mappings,
    build_aggregated_correlation_matrix,
    build_lower_triangle_matrix,
    build_pair_metric_matrix,
    build_metrics_only_results_df,
    compute_annotator_agreement_across_scales,
    compute_scale_correlations_by_rater,
    infer_label_type_for_columns,
    parse_annotation_column_pattern,
)


class MetricsOnlyEvaluationTests(unittest.TestCase):
    def test_parse_annotation_column_pattern_matches_default_template(self) -> None:
        parse_result = parse_annotation_column_pattern(
            ["Rater_chloe_clarity", "Rater_oli_usefulness"],
            "Rater_{rater}_{scale}",
        )

        self.assertIsNone(parse_result["error"])
        self.assertTrue(parse_result["supports_rater"])
        self.assertEqual(
            parse_result["column_metadata"]["Rater_chloe_clarity"],
            {"rater": "chloe", "scale": "clarity"},
        )
        self.assertEqual(
            parse_result["column_metadata"]["Rater_oli_usefulness"],
            {"rater": "oli", "scale": "usefulness"},
        )

    def test_parse_annotation_column_pattern_matches_reversed_template(self) -> None:
        parse_result = parse_annotation_column_pattern(
            ["clarity_chloe", "usefulness_oli"],
            "{scale}_{rater}",
        )

        self.assertIsNone(parse_result["error"])
        self.assertEqual(
            parse_result["column_metadata"]["clarity_chloe"],
            {"scale": "clarity", "rater": "chloe"},
        )
        self.assertEqual(
            parse_result["column_metadata"]["usefulness_oli"],
            {"scale": "usefulness", "rater": "oli"},
        )

    def test_parse_annotation_column_pattern_matches_dash_template(self) -> None:
        parse_result = parse_annotation_column_pattern(
            ["judge-chloe-clarity", "judge-oli-clarity"],
            "judge-{rater}-{scale}",
        )

        self.assertIsNone(parse_result["error"])
        self.assertEqual(
            parse_result["matched_scale_groups"]["clarity"],
            ["judge-chloe-clarity", "judge-oli-clarity"],
        )

    def test_parse_annotation_column_pattern_rejects_invalid_templates(self) -> None:
        parse_result = parse_annotation_column_pattern(
            ["clarity_chloe"],
            "{rater}_{dimension}",
        )

        self.assertIsNotNone(parse_result["error"])
        self.assertEqual(parse_result["matched_columns"], [])
        self.assertEqual(parse_result["unmatched_columns"], ["clarity_chloe"])

    def test_autodetect_scale_mappings_groups_rater_columns_by_scale(self) -> None:
        results_df = pd.DataFrame(
            {
                "Rater_chloe_clarity": [1, 2],
                "Rater_oli_clarity": [1, 2],
                "Rater_julien_clarity": [2, 2],
                "Rater_chloe_usefulness": [0, 1],
                "Rater_oli_usefulness": [0, 1],
                "Rater_julien_usefulness": [1, 1],
                "Other_column": ["x", "y"],
            }
        )

        mappings = autodetect_scale_mappings(
            results_df,
            [
                "Rater_chloe_clarity",
                "Rater_oli_clarity",
                "Rater_julien_clarity",
                "Rater_chloe_usefulness",
                "Rater_oli_usefulness",
                "Rater_julien_usefulness",
            ],
        )

        self.assertEqual(
            [mapping["name"] for mapping in mappings], ["clarity", "usefulness"]
        )
        self.assertEqual(
            mappings[0]["annotation_columns"],
            [
                "Rater_chloe_clarity",
                "Rater_julien_clarity",
                "Rater_oli_clarity",
            ],
        )
        self.assertEqual(
            mappings[0]["human_columns"],
            mappings[0]["annotation_columns"],
        )
        self.assertEqual(mappings[0]["label_type"], "Integer")
        self.assertTrue(
            mappings[0]["llm_field"].startswith("__metrics_only_prediction_")
        )
        self.assertEqual(
            mappings[0]["kripp_level_of_measurement"],
            "ordinal",
        )

    def test_autodetect_scale_mappings_supports_custom_templates(self) -> None:
        results_df = pd.DataFrame(
            {
                "clarity_chloe": [1, 2],
                "clarity_oli": [1, 2],
                "usefulness_chloe": [0, 1],
                "usefulness_oli": [0, 1],
            }
        )
        parse_result = parse_annotation_column_pattern(
            list(results_df.columns),
            "{scale}_{rater}",
        )

        mappings = autodetect_scale_mappings(
            results_df,
            list(results_df.columns),
            column_metadata=parse_result["column_metadata"],
        )

        self.assertEqual(
            [mapping["name"] for mapping in mappings],
            ["clarity", "usefulness"],
        )
        self.assertEqual(
            mappings[0]["annotation_columns"],
            ["clarity_chloe", "clarity_oli"],
        )

    def test_autodetect_scale_mappings_keeps_matched_groups_and_lists_unmatched(
        self,
    ) -> None:
        results_df = pd.DataFrame(
            {
                "clarity_chloe": [1, 2],
                "clarity_oli": [1, 2],
                "notes": ["a", "b"],
            }
        )
        parse_result = parse_annotation_column_pattern(
            list(results_df.columns),
            "{scale}_{rater}",
        )

        mappings = autodetect_scale_mappings(
            results_df,
            list(results_df.columns),
            column_metadata=parse_result["column_metadata"],
        )

        self.assertEqual(parse_result["unmatched_columns"], ["notes"])
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0]["name"], "clarity")

    def test_autodetect_preserves_existing_label_type_and_correction(self) -> None:
        results_df = pd.DataFrame(
            {
                "Rater_chloe_clarity": [1, 2, 3],
                "Rater_oli_clarity": [1, 2, 2],
                "Rater_julien_clarity": [1, 3, 3],
                "Rater_chloe_open_endedness": ["short", "long", "medium"],
                "Rater_oli_open_endedness": ["short", "long", "medium"],
            }
        )

        mappings = autodetect_scale_mappings(
            results_df,
            list(results_df.columns),
            existing_mappings=[
                {
                    "id": "existing-clarity",
                    "name": "clarity",
                    "llm_field": "Rater_oli_clarity",
                    "human_columns": [
                        "Rater_chloe_clarity",
                        "Rater_julien_clarity",
                    ],
                    "annotation_columns": [
                        "Rater_chloe_clarity",
                        "Rater_oli_clarity",
                        "Rater_julien_clarity",
                    ],
                    "label_type": "Integer",
                    "kappa_weights": "quadratic",
                }
            ],
        )

        clarity_mapping = next(
            mapping for mapping in mappings if mapping["name"] == "clarity"
        )
        open_mapping = next(
            mapping for mapping in mappings if mapping["name"] == "open_endedness"
        )

        self.assertEqual(clarity_mapping["id"], "existing-clarity")
        self.assertEqual(
            clarity_mapping["llm_field"],
            "__metrics_only_prediction_existing-clarity",
        )
        self.assertEqual(clarity_mapping["kappa_weights"], "quadratic")
        self.assertEqual(open_mapping["label_type"], "Text")
        self.assertEqual(open_mapping["kripp_level_of_measurement"], "nominal")

    def test_build_metrics_only_results_df_derives_consensus_columns(self) -> None:
        results_df = pd.DataFrame(
            {
                "Rater_chloe_clarity": [1, 2, 1],
                "Rater_oli_clarity": [1, 2, 0],
                "Rater_julien_clarity": [0, 2, 1],
                "Rater_chloe_score": [0.5, 0.2, None],
                "Rater_oli_score": [0.7, 0.4, 0.6],
                "Rater_julien_score": [0.6, 0.3, 0.9],
            }
        )
        mappings = autodetect_scale_mappings(results_df, list(results_df.columns))

        derived_df = build_metrics_only_results_df(results_df, mappings)

        clarity_mapping = next(
            mapping for mapping in mappings if mapping["name"] == "clarity"
        )
        score_mapping = next(
            mapping for mapping in mappings if mapping["name"] == "score"
        )

        self.assertEqual(
            derived_df[clarity_mapping["llm_field"]].tolist(),
            [1, 2, 1],
        )
        self.assertEqual(score_mapping["label_type"], "Float")
        self.assertEqual(score_mapping["kripp_level_of_measurement"], "interval")
        self.assertEqual(
            derived_df[score_mapping["llm_field"]].round(4).tolist(),
            [0.6, 0.3, 0.75],
        )

    def test_classical_krippendorff_uses_annotation_columns_and_mapping_level(
        self,
    ) -> None:
        results_df = pd.DataFrame(
            {
                "a": [1, 2, 1],
                "b": [1, 2, 0],
                "c": [0, 2, 1],
            }
        )
        mapping: EvaluationMapping = {
            "id": "map1",
            "name": "Clarity",
            "llm_field": "__metrics_only_prediction_map1",
            "human_columns": ["a", "b", "c"],
            "annotation_columns": ["a", "b", "c"],
            "label_type": "Integer",
            "kripp_level_of_measurement": "ordinal",
        }
        derived_df = build_metrics_only_results_df(results_df, [mapping])

        class _FakeKrippendorff:
            @staticmethod
            def alpha(data, level_of_measurement):
                self.assertEqual(level_of_measurement, "ordinal")
                self.assertEqual(data.shape, (3, 3))
                return 0.42

        with patch.object(evaluation_dashboard, "krippendorff_lib", _FakeKrippendorff):
            result = compute_classical_krippendorff_results(
                derived_df,
                [mapping],
                level_of_measurement="nominal",
                use_mapping_levels=True,
            )[0]

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["alpha_overall"], 0.42)
        self.assertAlmostEqual(result["alpha_mean"], 0.42)

    def test_classical_krippendorff_requires_two_annotation_columns(self) -> None:
        results_df = pd.DataFrame({"a": [1, 2, 1]})
        mapping: EvaluationMapping = {
            "id": "map1",
            "name": "Clarity",
            "llm_field": "__metrics_only_prediction_map1",
            "human_columns": ["a"],
            "annotation_columns": ["a"],
            "label_type": "Integer",
            "kripp_level_of_measurement": "ordinal",
        }
        derived_df = build_metrics_only_results_df(results_df, [mapping])

        class _FakeKrippendorff:
            @staticmethod
            def alpha(data, level_of_measurement):
                return 0.42

        with patch.object(evaluation_dashboard, "krippendorff_lib", _FakeKrippendorff):
            result = compute_classical_krippendorff_results(
                derived_df,
                [mapping],
                level_of_measurement="ordinal",
                use_mapping_levels=True,
            )[0]

        self.assertEqual(result["status"], "skipped")
        self.assertIn("at least 2 annotation columns", result["reason"])

    def test_compute_scale_correlations_by_rater_groups_scales_per_annotator(
        self,
    ) -> None:
        results_df = pd.DataFrame(
            {
                "Rater_chloe_clarity": [1, 2, 3, 4],
                "Rater_chloe_usefulness": [2, 4, 6, 8],
                "Rater_chloe_complexity": [4, 3, 2, 1],
                "Rater_oli_clarity": [1, 1, 2, 2],
                "Rater_oli_usefulness": [1, 2, 2, 3],
                "Rater_oli_complexity": [3, 2, 2, 1],
                "Rater_chloe_open_endedness": ["low", "mid", "high", "high"],
                "Rater_oli_open_endedness": ["low", "mid", "mid", "high"],
            }
        )
        mappings = autodetect_scale_mappings(results_df, list(results_df.columns))

        correlation_results = compute_scale_correlations_by_rater(
            results_df,
            mappings,
            method="spearman",
            min_overlap=3,
        )

        self.assertEqual(
            correlation_results["skipped_non_numeric"],
            ["open_endedness"],
        )
        self.assertIn("chloe", correlation_results["by_rater"])
        self.assertIn("oli", correlation_results["by_rater"])

        chloe_pairs = correlation_results["by_rater"]["chloe"]["pairwise_df"]
        top_pair = chloe_pairs.iloc[0]
        self.assertIn(top_pair["Scale A"], {"clarity", "complexity"})
        self.assertIn(top_pair["Scale B"], {"clarity", "usefulness", "complexity"})
        self.assertAlmostEqual(abs(float(top_pair["Correlation"])), 1.0)

        summary_df = correlation_results["summary_df"]
        self.assertEqual(summary_df.iloc[0]["Rater"], "chloe")
        self.assertGreaterEqual(float(summary_df.iloc[0]["Max |correlation|"]), 1.0)

        strongest_pairs_overall_df = correlation_results["strongest_pairs_overall_df"]
        top_overall_pair = strongest_pairs_overall_df.iloc[0]
        self.assertIn(
            {top_overall_pair["Scale A"], top_overall_pair["Scale B"]},
            [
                {"clarity", "complexity"},
                {"clarity", "usefulness"},
                {"complexity", "usefulness"},
            ],
        )
        self.assertEqual(int(top_overall_pair["Raters"]), 2)
        self.assertAlmostEqual(abs(float(top_overall_pair["Average correlation"])), 1.0)

        aggregate_abs_matrix = build_aggregated_correlation_matrix(
            strongest_pairs_overall_df,
            value_column="Average |correlation|",
        )
        self.assertEqual(
            sorted(aggregate_abs_matrix.index.tolist()),
            ["clarity", "complexity", "usefulness"],
        )
        self.assertAlmostEqual(
            float(aggregate_abs_matrix.loc["clarity", "clarity"]), 1.0
        )
        self.assertAlmostEqual(
            float(aggregate_abs_matrix.loc["clarity", "usefulness"]),
            0.8535533905932738,
        )

        triangular_matrix = build_lower_triangle_matrix(aggregate_abs_matrix)
        self.assertTrue(pd.isna(triangular_matrix.loc["clarity", "usefulness"]))
        self.assertAlmostEqual(
            float(triangular_matrix.loc["usefulness", "clarity"]),
            0.8535533905932738,
        )

    def test_compute_scale_correlations_by_rater_supports_custom_pattern_metadata(
        self,
    ) -> None:
        results_df = pd.DataFrame(
            {
                "clarity_chloe": [1, 2, 3, 4],
                "usefulness_chloe": [2, 4, 6, 8],
                "complexity_chloe": [4, 3, 2, 1],
                "clarity_oli": [1, 1, 2, 2],
                "usefulness_oli": [1, 2, 2, 3],
                "complexity_oli": [3, 2, 2, 1],
            }
        )
        parse_result = parse_annotation_column_pattern(
            list(results_df.columns),
            "{scale}_{rater}",
        )
        mappings = autodetect_scale_mappings(
            results_df,
            list(results_df.columns),
            column_metadata=parse_result["column_metadata"],
        )

        correlation_results = compute_scale_correlations_by_rater(
            results_df,
            mappings,
            column_metadata=parse_result["column_metadata"],
            method="spearman",
            min_overlap=3,
        )

        self.assertIn("chloe", correlation_results["by_rater"])
        self.assertIn("oli", correlation_results["by_rater"])
        self.assertTrue(correlation_results["summary_df"].shape[0] >= 2)

    def test_compute_annotator_agreement_across_scales_ranks_closest_peer(self) -> None:
        results_df = pd.DataFrame(
            {
                "Rater_chloe_clarity": [1, 1, 2, 2, 3],
                "Rater_julien_clarity": [1, 1, 2, 2, 3],
                "Rater_oli_clarity": [0, 1, 2, 2, 1],
                "Rater_chloe_usefulness": [0, 1, 1, 2, 2],
                "Rater_julien_usefulness": [0, 1, 1, 2, 2],
                "Rater_oli_usefulness": [2, 2, 1, 0, 0],
                "Rater_chloe_open_endedness": ["low", "mid", "high", "high", "mid"],
                "Rater_julien_open_endedness": ["low", "mid", "high", "high", "mid"],
                "Rater_oli_open_endedness": ["mid", "mid", "mid", "low", "low"],
            }
        )
        mappings = autodetect_scale_mappings(results_df, list(results_df.columns))

        agreement_results = compute_annotator_agreement_across_scales(
            results_df,
            mappings,
            min_overlap=2,
        )

        self.assertIn(
            "chloe", agreement_results["annotator_summary_df"]["Annotator"].tolist()
        )
        self.assertIn(
            "julien", agreement_results["annotator_summary_df"]["Annotator"].tolist()
        )
        self.assertIn(
            "oli", agreement_results["annotator_summary_df"]["Annotator"].tolist()
        )

        top_summary_row = agreement_results["annotator_summary_df"].iloc[0]
        self.assertIn(top_summary_row["Annotator"], {"chloe", "julien"})
        self.assertIn(top_summary_row["Closest peer"], {"chloe", "julien"})
        self.assertGreater(float(top_summary_row["Average agreement to others"]), 0.0)

        aggregated_pairs_df = agreement_results["aggregated_pairs_df"]
        top_pair_row = aggregated_pairs_df.iloc[0]
        self.assertEqual(
            {top_pair_row["Annotator A"], top_pair_row["Annotator B"]},
            {"chloe", "julien"},
        )

        per_scale_df = agreement_results["per_scale_df"]
        clarity_chloe_oli = per_scale_df[
            (per_scale_df["Scale"] == "clarity")
            & (per_scale_df["Annotator A"] == "chloe")
            & (per_scale_df["Annotator B"] == "oli")
        ].iloc[0]
        self.assertEqual(int(clarity_chloe_oli["Disagreements"]), 2)
        self.assertAlmostEqual(float(clarity_chloe_oli["Conflict %"]), 0.4)

        annotator_matrix = build_pair_metric_matrix(
            aggregated_pairs_df,
            left_col="Annotator A",
            right_col="Annotator B",
            value_col="Average agreement",
        )
        self.assertAlmostEqual(float(annotator_matrix.loc["chloe", "chloe"]), 1.0)
        self.assertGreater(
            float(annotator_matrix.loc["chloe", "julien"]),
            float(annotator_matrix.loc["chloe", "oli"]),
        )

    def test_compute_annotator_agreement_supports_custom_pattern_metadata(self) -> None:
        results_df = pd.DataFrame(
            {
                "clarity_chloe": [1, 1, 2, 2],
                "clarity_julien": [1, 1, 2, 2],
                "clarity_oli": [0, 1, 2, 1],
                "usefulness_chloe": [0, 1, 1, 2],
                "usefulness_julien": [0, 1, 1, 2],
                "usefulness_oli": [2, 2, 1, 0],
            }
        )
        parse_result = parse_annotation_column_pattern(
            list(results_df.columns),
            "{scale}_{rater}",
        )
        mappings = autodetect_scale_mappings(
            results_df,
            list(results_df.columns),
            column_metadata=parse_result["column_metadata"],
        )

        agreement_results = compute_annotator_agreement_across_scales(
            results_df,
            mappings,
            column_metadata=parse_result["column_metadata"],
            min_overlap=2,
        )

        self.assertIn(
            "chloe", agreement_results["annotator_summary_df"]["Annotator"].tolist()
        )
        self.assertIn(
            "julien", agreement_results["annotator_summary_df"]["Annotator"].tolist()
        )
        self.assertIn(
            "oli", agreement_results["annotator_summary_df"]["Annotator"].tolist()
        )

    def test_rater_aware_analyses_require_rater_metadata(self) -> None:
        results_df = pd.DataFrame(
            {
                "clarity_a": [1, 2, 3],
                "usefulness_a": [2, 3, 4],
                "clarity_b": [1, 1, 2],
                "usefulness_b": [3, 2, 1],
            }
        )
        mappings: list[EvaluationMapping] = [
            {
                "id": "clarity",
                "name": "clarity",
                "llm_field": "__metrics_only_prediction_clarity",
                "human_columns": ["clarity_a", "clarity_b"],
                "annotation_columns": ["clarity_a", "clarity_b"],
                "label_type": "Integer",
            },
            {
                "id": "usefulness",
                "name": "usefulness",
                "llm_field": "__metrics_only_prediction_usefulness",
                "human_columns": ["usefulness_a", "usefulness_b"],
                "annotation_columns": ["usefulness_a", "usefulness_b"],
                "label_type": "Integer",
            },
        ]
        column_metadata = {
            "clarity_a": {"scale": "clarity"},
            "clarity_b": {"scale": "clarity"},
            "usefulness_a": {"scale": "usefulness"},
            "usefulness_b": {"scale": "usefulness"},
        }

        correlation_results = compute_scale_correlations_by_rater(
            results_df,
            mappings,
            column_metadata=column_metadata,
            method="spearman",
            min_overlap=2,
        )
        agreement_results = compute_annotator_agreement_across_scales(
            results_df,
            mappings,
            column_metadata=column_metadata,
            min_overlap=2,
        )

        self.assertTrue(correlation_results["summary_df"].empty)
        self.assertEqual(
            sorted(correlation_results["unmatched_columns"]),
            ["clarity_a", "clarity_b", "usefulness_a", "usefulness_b"],
        )
        self.assertTrue(agreement_results["annotator_summary_df"].empty)
        self.assertEqual(
            sorted(agreement_results["unmatched_columns"]),
            ["clarity_a", "clarity_b", "usefulness_a", "usefulness_b"],
        )

    def test_infer_label_type_detects_text_and_float(self) -> None:
        results_df = pd.DataFrame(
            {
                "float_a": [1.5, 2.0, None],
                "float_b": [1.0, 2.5, 3.5],
                "text_a": ["low", "high", None],
                "text_b": ["medium", "high", "low"],
            }
        )

        self.assertEqual(
            infer_label_type_for_columns(results_df, ["float_a", "float_b"]),
            "Float",
        )
        self.assertEqual(
            infer_label_type_for_columns(results_df, ["text_a", "text_b"]),
            "Text",
        )

    def test_direct_metrics_work_without_step6_metadata(self) -> None:
        results_df = pd.DataFrame(
            {
                "prediction": [1, 0, 1, 0],
                "human_a": [1, 0, 1, 1],
                "human_b": [1, 0, 0, 0],
            }
        )
        mapping: EvaluationMapping = {
            "id": "map1",
            "name": "Clarity",
            "llm_field": "prediction",
            "human_columns": ["human_a", "human_b"],
            "label_type": "Integer",
        }

        classification_result = compute_classification_results(results_df, [mapping])[0]
        kappa_result = compute_kappa_results(results_df, [mapping])[0]

        self.assertEqual(classification_result["status"], "ok")
        self.assertFalse(classification_result["summary_df"].empty)
        self.assertEqual(int(classification_result["result_row"]["N_train"]), 4)
        self.assertIn("run", classification_result["prepared"].analysis_data.columns)
        self.assertIn("split", classification_result["prepared"].analysis_data.columns)

        self.assertEqual(kappa_result["status"], "ok")
        self.assertFalse(kappa_result["summary_df"].empty)
        self.assertIn("kappa_GT_train", kappa_result["result_row"])

    def test_multiple_mappings_support_different_label_types(self) -> None:
        results_df = pd.DataFrame(
            {
                "clarity_prediction": [1, 0, 1],
                "clarity_a": [1, 0, 1],
                "clarity_b": [1, 1, 1],
                "tone_prediction": ["positive", "negative", "positive"],
                "tone_a": ["positive", "negative", "neutral"],
                "tone_b": ["positive", "negative", "positive"],
            }
        )
        mappings: list[EvaluationMapping] = [
            {
                "id": "map1",
                "name": "Clarity",
                "llm_field": "clarity_prediction",
                "human_columns": ["clarity_a", "clarity_b"],
                "label_type": "Integer",
            },
            {
                "id": "map2",
                "name": "Tone",
                "llm_field": "tone_prediction",
                "human_columns": ["tone_a", "tone_b"],
                "label_type": "Text",
            },
        ]

        results = compute_classification_results(results_df, mappings)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["summary_df"].empty)
            self.assertGreater(len(result["prepared"].analysis_data), 0)

    def test_existing_run_column_is_aggregated_across_runs(self) -> None:
        results_df = pd.DataFrame(
            {
                "prompt_name": ["prompt"] * 6,
                "iteration": [1] * 6,
                "split": ["train"] * 6,
                "run": [1, 1, 1, 2, 2, 2],
                "prediction": [1, 0, 1, 1, 0, 1],
                "human_a": [1, 0, 1, 1, 0, 1],
                "human_b": [1, 1, 1, 1, 0, 0],
            }
        )
        mapping: EvaluationMapping = {
            "id": "map1",
            "name": "Clarity",
            "llm_field": "prediction",
            "human_columns": ["human_a", "human_b"],
            "label_type": "Integer",
        }

        classification_result = compute_classification_results(results_df, [mapping])[0]
        kappa_result = compute_kappa_results(results_df, [mapping])[0]

        self.assertTrue(classification_result["has_multiple_runs"])
        self.assertEqual(int(classification_result["result_row"]["n_runs"]), 2)
        self.assertTrue(kappa_result["has_multiple_runs"])
        self.assertEqual(int(kappa_result["result_row"]["n_runs"]), 2)


if __name__ == "__main__":
    unittest.main()
