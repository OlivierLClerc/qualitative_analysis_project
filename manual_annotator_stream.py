import streamlit as st
import pandas as pd
import io
from transformers import pipeline
from qualitative_analysis import load_data

# Inject CSS to adjust overall layout, radio buttons, and spacing
st.markdown(
    """
    <style>
    /* Adjust the main container margins and max-width */
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1200px;
    }
    /* Force the radiogroup container to display radio options horizontally */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        margin-bottom: 0.1rem;
    }
    /* Reduce spacing between radio options */
    div[role="radiogroup"] label {
        margin-right: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_translator():
    """Translator pipeline for FR -> EN."""
    return pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")


def load_codebook(file) -> str:
    """
    Loads the codebook text from a TXT file and returns it as a single string.
    """
    return file.read().decode("utf-8")


def main():
    st.title("Manual Annotation Tool")

    st.markdown(
        """
        This application is designed to help you **manually annotate** a dataset.
        Your dataset should be in CSV format, and each row should correspond to a single item to be annotated.
        You will be able to load your codebook (instructions) and define the labels you want to use.
        The tool will guide you through each row, allowing you to annotate them one by one.
        **Unvalid** rows can be flagged (it will create a new column for each annotator).
        
        **Steps**  
        1. **Upload Data** (CSV)  
        2. **Set Annotator Name** (creates a new column for your annotations)  
        3. **Upload Codebook** (TXT) to display instructions on the sidebar (optional)  
        4. **Define Fast Labels**  
        5. **Select Columns to Display**  
        6. **Annotate Row-by-Row**  
        7. **Download Updated Data**  
        """
    )

    # -------------------------------
    # SESSION STATE INIT
    # -------------------------------
    if "df" not in st.session_state:
        st.session_state.df = None
    if "translated_rows" not in st.session_state:
        st.session_state.translated_rows = {}
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "annotator_name" not in st.session_state:
        st.session_state.annotator_name = ""
    if "new_col_name" not in st.session_state:
        st.session_state.new_col_name = ""
    # For the read-only codebook text
    if "codebook_text" not in st.session_state:
        st.session_state.codebook_text = ""
    # For fast label definitions
    if "fast_labels_text" not in st.session_state:
        st.session_state.fast_labels_text = ""
    if "fast_label" not in st.session_state:
        st.session_state.fast_label = ""

    # --------------------------------
    # STEP 1: Upload CSV Dataset
    # --------------------------------
    st.header("Step 1: Upload Your Data (CSV)")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="data_file")

    if uploaded_file is not None and st.session_state.df is None:
        # Load data via your custom function from qualitative_analysis
        st.session_state.df = load_data(
            uploaded_file, file_type="csv", delimiter=";"
        ).reset_index(drop=True)
        st.success(f"Data loaded successfully ({len(st.session_state.df)} rows).")

    if st.session_state.df is None:
        st.stop()

    df = st.session_state.df

    # --------------------------------
    # STEP 2: Set Annotator Name
    # --------------------------------
    st.header("Step 2: Set Annotator Name")

    annotator = st.text_input(
        "Annotator Name / Initials:",
        value=st.session_state.annotator_name,
        key="annotator_input",
    )

    st.markdown(
        "A new column will be created for your annotations if it doesn't exist."
    )

    if annotator:
        st.session_state.annotator_name = annotator

    # Create or re-use the column if user clicks confirm
    if st.session_state.new_col_name == "":
        if st.button("Confirm Annotator Name"):
            if st.session_state.annotator_name:
                candidate_col = f"Rater_{st.session_state.annotator_name}"
                flag_col = f"Unvalid_{st.session_state.annotator_name}"
                if candidate_col not in df.columns:
                    df[candidate_col] = pd.NA
                if flag_col not in df.columns:
                    df[flag_col] = False
                st.session_state.new_col_name = candidate_col
                st.success(f"Created/Found columns '{candidate_col}' and '{flag_col}'!")
                st.rerun()

    if not st.session_state.new_col_name:
        st.stop()

    # --------------------------------
    # STEP 3: Upload Codebook (Optional)
    # --------------------------------
    st.header("Step 3: Upload Codebook (Optional)")

    st.markdown(
        "You may upload a TXT file with classification instructions. "
        "Its contents will be displayed on the sidebar while you annotate."
    )

    codebook_file = st.file_uploader(
        "Upload Codebook TXT file:", type=["txt"], key="codebook_file"
    )

    if codebook_file is not None:
        codebook_str = load_codebook(codebook_file)
        st.session_state.codebook_text = codebook_str
        st.success("Codebook loaded successfully!")

    # If a codebook has been loaded, show it in the sidebar
    if st.session_state.codebook_text:
        st.sidebar.header("Codebook Instructions")
        formatted_text = st.session_state.codebook_text.replace("\n", "<br>")
        st.sidebar.markdown(formatted_text, unsafe_allow_html=True)

    # --------------------------------
    # STEP 4: Define Labels
    # --------------------------------
    st.header("Step 4: Define Labels")

    st.markdown(
        "Define the possible labels of annotations that you can apply. "
        "Simply type them in a comma-separated list."
    )

    st.text_input(
        "Labels (comma-separated):",
        key="fast_labels_text",
        placeholder="e.g. 0, 1, or cat, dog, etc.",
    )

    # --------------------------------
    # STEP 5: Select Columns to Display
    # --------------------------------
    st.header("Step 5: Select Columns to Display")

    flag_col = f"Unvalid_{st.session_state.annotator_name}"
    possible_columns = [
        c for c in df.columns if c not in (st.session_state.new_col_name, flag_col)
    ]

    selected_columns = st.multiselect(
        "Choose which columns to see while annotating:",
        options=possible_columns,
        key="selected_columns",
    )

    if not selected_columns:
        st.info("No columns selected. Please pick at least one.")
        st.stop()

    # --------------------------------
    # STEP 6: Annotate Row-by-Row
    # --------------------------------
    st.header("Step 6: Annotate Rows")

    idx = st.session_state.current_index
    # Ensure index is in valid range
    if idx < 0:
        st.session_state.current_index = 0
        idx = 0
    if idx >= len(df):
        st.session_state.current_index = len(df) - 1
        idx = len(df) - 1

    # Show existing rating & flagged status
    rating_val = df.at[idx, st.session_state.new_col_name]
    flagged_val = df.at[idx, flag_col] if flag_col in df.columns else None

    st.markdown(f"**Row Index:** {idx}")
    st.markdown(f"**Existing Rating:** {rating_val}")
    st.markdown(f"**Is Unvalid:** {flagged_val}")

    # Display the selected columns
    for col in selected_columns:
        st.write(f"**{col}:** {df.at[idx, col]}")

    # Translation (optional)
    translate_row = st.checkbox("Translate this row to English", key="translate_row")
    if translate_row:
        if idx not in st.session_state.translated_rows:
            translator = get_translator()
            translation_dict = {}
            for col in selected_columns:
                text = str(df.at[idx, col])
                try:
                    result = translator(text)
                    translation_dict[col] = result[0]["translation_text"]
                except Exception as e:
                    st.error(f"Error translating '{col}': {e}")
                    translation_dict[col] = "[Error]"
            st.session_state.translated_rows[idx] = translation_dict

        # Display the stored translation
        translations = st.session_state.translated_rows[idx]
        st.markdown("### Translated Content:")
        for col, tval in translations.items():
            st.write(f"**{col}:** {tval}")

    # Navigation Buttons
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Previous"):
            # Apply the currently selected fast label if any
            if st.session_state.fast_label != "":
                df.at[idx, st.session_state.new_col_name] = st.session_state.fast_label

            # Reset fast_label and go previous
            st.session_state.fast_label = ""
            st.session_state.current_index = max(0, idx - 1)
            st.rerun()

    with c2:
        if st.button("Next"):
            # Apply the currently selected fast label if any
            if st.session_state.fast_label != "":
                df.at[idx, st.session_state.new_col_name] = st.session_state.fast_label

            # Reset fast_label and go next
            st.session_state.fast_label = ""
            st.session_state.current_index = min(len(df) - 1, idx + 1)
            st.rerun()

    with c3:
        if st.button("Unvalid data"):
            # Mark row as unvalid
            df.at[idx, flag_col] = True
            # Also apply fast_label if set
            if st.session_state.fast_label != "":
                df.at[idx, st.session_state.new_col_name] = st.session_state.fast_label

            # Reset fast_label and go next
            st.session_state.fast_label = ""
            st.session_state.current_index = min(len(df) - 1, idx + 1)
            st.rerun()

    # Fast Label (below data)
    fast_labels = [
        label.strip()
        for label in st.session_state.fast_labels_text.split(",")
        if label.strip()
    ]
    if fast_labels:
        selected_fast_label = st.radio(
            "Select a Label:", options=[""] + fast_labels, key="fast_label"
        )
        if selected_fast_label:
            st.markdown(f"**Current Label:** {selected_fast_label}")

    # --------------------------------
    # STEP 7: Download Updated Data
    # --------------------------------
    st.header("Step 7: Download Updated Data")
    st.markdown(
        "When you're done (or want to pause), download your annotated data as an Excel file."
    )

    filename_input = st.text_input(
        "Output filename:", value="annotated_data.xlsx", key="results_filename_input"
    )
    if not filename_input.endswith(".xlsx"):
        filename_input += ".xlsx"

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=filename_input,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
