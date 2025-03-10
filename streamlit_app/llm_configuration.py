"""
Module for handling LLM configuration functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, Optional

from qualitative_analysis import get_llm_client
import qualitative_analysis.config as config


def configure_llm(app_instance: Any) -> Optional[Any]:
    """
    Step 5: Choose the Model
    Lets the user pick the LLM provider, supply an API key (if not in .env),
    and choose a model.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        The LLM client or None if configuration is incomplete
    """
    st.markdown("### Step 5: Choose the Model", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 5", expanded=True):
        # 🚨 Block Step 5 if Step 4 is incomplete
        if not app_instance.selected_fields:
            st.warning(
                "⚠️ Please specify at least one field to extract in Step 4 before continuing."
            )
            return None

        provider_options = ["Select Provider", "OpenAI", "Together", "Azure"]
        selected_provider_display = st.selectbox(
            "Select LLM Provider:", provider_options, key="llm_provider_select"
        )

        if selected_provider_display == "Select Provider":
            st.info("ℹ️ Please select a provider to continue.")
            return None

        provider_map = {"OpenAI": "openai", "Together": "together", "Azure": "azure"}
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
                "together": "together-...",
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

        # Select model
        if selected_provider_display == "OpenAI":
            model_options = ["gpt-4o", "gpt-4o-mini"]
        elif selected_provider_display == "Together":
            model_options = ["gpt-neoxt-chat-20B"]
        else:  # Azure
            model_options = ["gpt-4o", "gpt-4o-mini"]

        chosen_model = st.selectbox(
            "Select Model:",
            model_options,
            key="llm_model_select",
        )

        app_instance.selected_model = chosen_model
        st.session_state["selected_model"] = chosen_model

    return app_instance.llm_client
