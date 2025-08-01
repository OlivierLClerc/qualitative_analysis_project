# __init__.py
from .data_processing import (
    load_data,
    clean_and_normalize,
    sanitize_dataframe,
    select_and_rename_columns,
    load_results_from_csv,
)
from .metrics import (
    compute_cohens_kappa,
    compute_all_kappas,
    plot_confusion_matrices,
    compute_classification_metrics,
    run_alt_test_general,
    compute_detailed_kappa_metrics,
    compute_classification_metrics_from_results,
)
from .model_interaction import (
    LLMClient,
    OpenAILLMClient,
    TogetherLLMClient,
    get_llm_client,
)
from .notebooks_functions import (
    process_general_verbatims,
)
from .parsing import (
    parse_llm_response,
    extract_code_from_response,
    extract_global_validity,
)
from .cost_estimation import openai_api_calculate_cost
from .logging import calculate_and_log
from .scenario_runner import run_scenarios

__all__ = [
    # Data processing
    "load_data",
    "clean_and_normalize",
    "sanitize_dataframe",
    "select_and_rename_columns",
    "load_results_from_csv",
    # Metrics
    "compute_cohens_kappa",
    "compute_all_kappas",
    "compute_classification_metrics",
    "compute_classification_metrics_from_results",
    "compute_detailed_kappa_metrics",
    "plot_confusion_matrices",
    "run_alt_test_general",
    # Model interaction
    "LLMClient",
    "OpenAILLMClient",
    "TogetherLLMClient",
    "get_llm_client",
    # Notebooks functions
    "process_general_verbatims",
    # Parsing
    "parse_llm_response",
    "extract_code_from_response",
    "extract_global_validity",
    # Prompt construction
    "build_data_format_description",
    "construct_prompt",
    # Cost estimation
    "openai_api_calculate_cost",
    # Logging
    "calculate_and_log",
    # Scenario runner
    "run_scenarios",
]
