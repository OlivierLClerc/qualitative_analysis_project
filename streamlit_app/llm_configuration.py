"""
Module for handling LLM configuration functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, Optional

from qualitative_analysis import get_llm_client
import qualitative_analysis.config as config


def configure_llm(
    app_instance: Any, step_number: int = 5, purpose: str = ""
) -> Optional[Any]:
    """
    Choose the Model
    Lets the user pick the LLM provider, supply an API key (if not in .env),
    and choose a model.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=5 for annotation mode)
        purpose: Optional purpose label (e.g., "for Generation", "for Annotation")

    Returns:
        The LLM client or None if configuration is incomplete
    """
    title = f"### Step {step_number}: Choose the Model"
    if purpose:
        title += f" {purpose}"
    st.markdown(title, unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        # Dependency validation based on context
        # For annotation mode (step_number=5), check if fields are defined
        if step_number == 5 and not app_instance.selected_fields:
            st.warning(
                "⚠️ Please specify at least one field to extract in Step 4 before continuing."
            )
            return None

        # For generation mode step 3 (generation LLM), check if generation config exists
        if step_number == 3 and purpose == "for Generation":
            if (
                not hasattr(app_instance, "generation_config")
                or not app_instance.generation_config
            ):
                st.warning(
                    "⚠️ Please configure generation settings in Step 2 before continuing."
                )
                return None

        # For generation mode step 8 (annotation LLM), check if codebook and fields are defined
        if step_number == 8 and purpose == "for Annotation":
            if not app_instance.codebook or not app_instance.codebook.strip():
                st.warning("⚠️ Please provide a codebook in Step 6 before continuing.")
                return None
            if not app_instance.selected_fields:
                st.warning(
                    "⚠️ Please specify fields to extract in Step 7 before continuing."
                )
                return None

        provider_options = [
            "Select Provider",
            "OpenAI",
            "Anthropic",
            "Gemini",
            "Together",
            "OpenRouter",
            "Azure",
        ]

        # Make key unique based on step number
        provider_key = f"llm_provider_select_step{step_number}"

        selected_provider_display = st.selectbox(
            "Select LLM Provider:", provider_options, key=provider_key
        )

        if selected_provider_display == "Select Provider":
            st.info("ℹ️ Please select a provider to continue.")
            return None

        provider_map = {
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Gemini": "gemini",
            "Together": "together",
            "OpenRouter": "openrouter",
            "Azure": "azure",
        }
        internal_provider = provider_map[selected_provider_display]

        # Check config for an existing API key
        existing_api_key = config.MODEL_CONFIG[internal_provider].get("api_key")

        if existing_api_key:
            st.success(f"🔑 API Key loaded from .env for {selected_provider_display}!")
            final_api_key = existing_api_key
        else:
            st.sidebar.subheader("API Key Configuration")
            api_key_placeholder = {
                "openai": "sk-...",
                "anthropic": "sk-ant-...",
                "gemini": "your-gemini-api-key",
                "together": "together-...",
                "openrouter": "sk-or-...",
                "azure": "azure-...",
            }.get(internal_provider, "Enter API Key")

            api_key = st.sidebar.text_input(
                f"Enter your {selected_provider_display} API Key",
                type="password",
                placeholder=api_key_placeholder,
            )
            st.sidebar.info(
                "🔒 Your API key is used only during this session and is never stored."
            )

            if not api_key:
                st.warning(f"Please provide your {selected_provider_display} API key.")
                st.stop()
            else:
                st.success(f"{selected_provider_display} API Key provided!")
                final_api_key = api_key

        # Update config with the final API key
        provider_config = config.MODEL_CONFIG[internal_provider].copy()
        provider_config["api_key"] = final_api_key

        # Instantiate the LLM client
        app_instance.llm_client = get_llm_client(
            provider=internal_provider, config=provider_config
        )

        # Select model - make keys unique based on step number
        model_key = f"llm_model_select_step{step_number}"

        if selected_provider_display == "OpenRouter":
            # For OpenRouter, use text input to allow any model name
            st.markdown(
                """
                **OpenRouter Model Selection**
                
                Enter the full model name in the format `provider/model-name`. 
                Visit [OpenRouter Models](https://openrouter.ai/models) to see all available models.
                """
            )

            chosen_model = st.text_input(
                "Enter OpenRouter Model Name:",
                value=st.session_state.get(
                    "selected_model", "anthropic/claude-3.5-sonnet"
                ),
                placeholder="anthropic/claude-3.5-sonnet",
                key=f"openrouter_model_input_step{step_number}",
                help="Format: provider/model-name (e.g., anthropic/claude-3.5-sonnet)",
            )

            if not chosen_model.strip():
                st.warning("Please enter a model name to continue.")
                return None

            if "/" not in chosen_model:
                st.warning(
                    "Model name should be in format 'provider/model-name' (e.g., 'anthropic/claude-3.5-sonnet')"
                )
                return None

        elif selected_provider_display == "OpenAI":
            model_options = ["gpt-4o", "gpt-4o-mini"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
        elif selected_provider_display == "Anthropic":
            model_options = ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
        elif selected_provider_display == "Gemini":
            model_options = ["gemini-2.0-flash-001", "gemini-2.5-pro-preview-03-25"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
        elif selected_provider_display == "Together":
            model_options = ["gpt-neoxt-chat-20B"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
        else:  # Azure
            model_options = ["gpt-4o", "gpt-4o-mini"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )

        app_instance.selected_model = chosen_model
        st.session_state["selected_model"] = chosen_model

    return app_instance.llm_client
