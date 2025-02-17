import streamlit as st
import pandas as pd
import json
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
    /* Force the radiogroup container to display radio options horizontally and reduce vertical spacing between options */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        margin-bottom: 0.05rem; /* reduces space between radio groups */
    }
    /* Reduce spacing between radio options */
    div[role="radiogroup"] label {
        margin-right: 0.3rem;
    }
    /* Reduce space between questions in the list (above the data) */
    .question {
        margin-bottom: 1rem;
    }
    /* Reduce vertical space between each radio widget (i.e. each question's radio control) */
    div[data-testid="stRadio"] {
        margin-bottom: 0.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_questions(file):
    """Load questions from a TXT, JSON, or Excel file.

    For TXT: one question per line.
    For JSON: expects a JSON array of strings.
    For Excel: takes the first column's nonempty values.
    """
    filename = file.name.lower()
    if filename.endswith(".txt"):
        content = file.read().decode("utf-8")
        questions = [line.strip() for line in content.splitlines() if line.strip()]
    elif filename.endswith(".json"):
        content = file.read().decode("utf-8")
        try:
            questions = json.loads(content)
            if not isinstance(questions, list):
                st.error("JSON file must contain an array of questions.")
                questions = []
        except Exception as e:
            st.error(f"Error reading JSON file: {e}")
            questions = []
    elif filename.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(file)
            questions = df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            questions = []
    else:
        st.error("Unsupported file format for questions.")
        questions = []
    return questions


# Cache the translator so that it is only loaded once.
@st.cache_resource(show_spinner=False)
def get_translator():
    return pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")


def save_current_answers(row_idx, answers_dict, warn_if_incomplete=False):
    """
    Saves the user's answers for the given row into the DataFrame.
    If a fast label is selected, it takes precedence over the question answers.
    """
    df = st.session_state.df
    new_col = st.session_state.new_col_name

    # Read the fast label value from its widget (key "fast_label")
    fast_label = st.session_state.get("fast_label", "")
    if fast_label != "":
        df.at[row_idx, new_col] = fast_label
        # Optionally, save individual question answers as well
        for q_idx, val in answers_dict.items():
            st.session_state.answers[(row_idx, q_idx)] = val
        return True

    # If no questions were loaded, there's nothing to do for the row
    if not st.session_state.questions:
        return True

    # Verify all answers are "yes" or "no"
    all_filled = all(a in ("yes", "no") for a in answers_dict.values())
    if not all_filled:
        if warn_if_incomplete:
            st.warning("Not all questions answered. Row not fully saved.")
        # Save partial answers to session, but do not finalize row
        for q_idx, val in answers_dict.items():
            st.session_state.answers[(row_idx, q_idx)] = val
        return False

    # If all "yes", set 1, otherwise 0
    all_yes = all(a == "yes" for a in answers_dict.values())
    df.at[row_idx, new_col] = 1 if all_yes else 0

    # Store answers in session state for reference
    for q_idx, val in answers_dict.items():
        st.session_state.answers[(row_idx, q_idx)] = val

    return True


def main():
    st.title("Data Rater Tool")

    # --- Initialize session state variables ---
    if "df" not in st.session_state:
        st.session_state.df = None
    if "translated_rows" not in st.session_state:
        st.session_state.translated_rows = {}
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "annotator_name" not in st.session_state:
        st.session_state.annotator_name = ""
    if "new_col_name" not in st.session_state:
        st.session_state.new_col_name = ""
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    # Fast labels definition input (do not use key "fast_label" here)
    if "fast_labels_text" not in st.session_state:
        st.session_state.fast_labels_text = ""

    # --- Step 1: File upload for data ---
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"], key="data_file")
    if uploaded_file is not None:
        if st.session_state.df is None:
            # Use the load_data function from the qualitative_analysis module
            st.session_state.df = load_data(
                uploaded_file, file_type="csv", delimiter=";"
            ).reset_index(drop=True)
        df = st.session_state.df

        # --- Step 2: Ask for annotator name ---
        annotator = st.text_input(
            "Annotator Name",
            value=st.session_state.annotator_name,
            key="annotator_input",
        )
        if annotator:
            st.session_state.annotator_name = annotator

        # Create the new column once the annotator name is confirmed.
        if st.session_state.new_col_name == "":
            if st.button("Confirm Annotator Name"):
                if st.session_state.annotator_name:
                    candidate_col = f"Rater_{st.session_state.annotator_name}"
                    flag_col = f"Unvalid_{st.session_state.annotator_name}"  # Create flagged column name
                    if candidate_col not in df.columns:
                        df[candidate_col] = pd.NA
                    if flag_col not in df.columns:
                        df[flag_col] = False  # Set default value to False
                    st.session_state.new_col_name = candidate_col
                    st.success(f"Columns '{candidate_col}' and '{flag_col}' created!")
                    st.rerun()

        # --- Optional: Load questions automatically from a file ---
        st.markdown("#### Add or load questions from file (TXT, JSON, Excel)")
        questions_file = st.file_uploader(
            "Upload Questions File",
            type=["txt", "json", "xlsx", "xls"],
            key="questions_file",
        )
        if questions_file is not None:
            loaded_questions = load_questions(questions_file)
            if loaded_questions:
                st.session_state.questions = loaded_questions
                st.success(f"Loaded {len(loaded_questions)} questions from file.")

        # --- Proceed only if the new rating column exists ---
        if st.session_state.new_col_name:
            # --- Step 3: Add questions manually (optional) ---
            new_question = st.text_input("Add a question?", key="new_question")
            if st.button("Add Question"):
                if new_question:
                    st.session_state.questions.append(new_question)
                    st.success(f"Added question: {new_question}")

            # --- Fast Labelling Definition ---
            st.markdown("### Fast Labelling Definition")
            st.text_input(
                "Define Fast Labels (comma separated):",
                key="fast_labels_text",
                placeholder="e.g., 0, 1 or cat, dog, bird",
            )

            # --- Step 4: Select columns to display (excluding rating and flagged columns) ---
            flag_col = f"Unvalid_{st.session_state.annotator_name}"
            possible_columns = [
                c
                for c in df.columns
                if c not in [st.session_state.new_col_name, flag_col]
            ]
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=possible_columns,
                key="selected_columns",
            )

            # --- Step 5: Display data row (main area) ---
            if selected_columns:
                idx = st.session_state.current_index
                if idx < 0:
                    st.session_state.current_index = 0
                    idx = 0
                if idx >= len(df):
                    st.session_state.current_index = len(df) - 1
                    idx = len(df) - 1

                # Display the annotator's rating
                rating_val = df.at[idx, st.session_state.new_col_name]
                st.markdown(f"**Current Index: {idx}**")
                st.markdown(f"**Existing Rating:** {rating_val}")

                # Display the flagged value (from the flagged column)
                flagged_val = df.at[idx, flag_col] if flag_col in df.columns else None
                st.markdown(f"**Is Unvalid:** {flagged_val}")

                # Then display the selected data columns
                for col in selected_columns:
                    st.write(f"**{col}:** {df.at[idx, col]}")

                # --- Row Translation Option ---
                translate_row = st.checkbox(
                    "Translate current row to English", key="translate_row"
                )
                if translate_row:
                    if idx not in st.session_state.translated_rows:
                        translator = get_translator()
                        translated = {}
                        for col in selected_columns:
                            text = str(df.at[idx, col])
                            try:
                                result = translator(text)
                                translated[col] = result[0]["translation_text"]
                            except Exception as e:
                                st.error(f"Error translating column '{col}': {e}")
                        st.session_state.translated_rows[idx] = translated
                    translated = st.session_state.translated_rows[idx]
                    st.markdown("### Translated Row:")
                    for col, trans in translated.items():
                        st.write(f"**{col}:** {trans}")

                # --- Navigation buttons (Previous, Next, Unvalid) ---
                col1, col2, col3 = st.columns(3)

                # --- Previous
                with col1:
                    if st.button("Previous"):
                        save_current_answers(idx, st.session_state.sidebar_answers)
                        st.session_state.current_index = max(0, idx - 1)
                        st.session_state.fast_label = ""
                        st.rerun()

                # --- Next
                with col2:
                    if st.button("Next"):
                        save_current_answers(idx, st.session_state.sidebar_answers)
                        st.session_state.current_index = min(len(df) - 1, idx + 1)
                        st.session_state.fast_label = ""
                        st.rerun()

                # --- Unvalid data
                with col3:
                    if st.button("Unvalid data"):
                        save_current_answers(idx, st.session_state.sidebar_answers)
                        # Set the flagged column to True
                        df.at[idx, flag_col] = True
                        # Move to the next row
                        st.session_state.current_index = min(len(df) - 1, idx + 1)
                        st.session_state.fast_label = ""
                        st.rerun()

                # --- Fast Label Selection (Displayed Under the Data) ---
                fast_labels = [
                    label.strip()
                    for label in st.session_state.get("fast_labels_text", "").split(",")
                    if label.strip()
                ]
                if fast_labels:
                    selected_fast_label = st.radio(
                        "Select Fast Label (overrides questions):",
                        options=[""] + fast_labels,
                        key="fast_label",
                    )
                    if selected_fast_label:
                        st.markdown(f"**Fast Label Selected:** {selected_fast_label}")

            # --- Step 6: Download Excel with Dynamic Filename ---
            st.write("**When done, you can download the updated data as Excel:**")

            # 1) Ask user for a filename
            filename_input = st.text_input(
                "Enter a filename for your results:",
                value="annotated_data.xlsx",
                key="results_filename_input",
            )
            # Ensure ends with .xlsx
            if not filename_input.endswith(".xlsx"):
                filename_input += ".xlsx"

            # 2) Create in-memory BytesIO buffer
            excel_buffer = io.BytesIO()
            # 3) Write the DataFrame to this buffer as an Excel file
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                st.session_state.df.to_excel(writer, index=False)

            # 4) Provide the download button for Excel file (using user-defined filename)
            st.download_button(
                label="Download Excel (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name=filename_input,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # --- Step 7: Show the sidebar only if there is at least one question ---
            if st.session_state.questions:
                st.sidebar.header("Annotation Questions")

                # Ensure we have a dict for the answers in the sidebar
                if "sidebar_answers" not in st.session_state:
                    st.session_state.sidebar_answers = {}

                # Display each question in the sidebar
                for i, question in enumerate(st.session_state.questions):
                    previous_choice = st.session_state.sidebar_answers.get(i, "")
                    options = ["", "yes", "no"]
                    default_index = (
                        options.index(previous_choice)
                        if previous_choice in options
                        else 0
                    )
                    answer = st.sidebar.radio(
                        label=question,
                        options=options,
                        index=default_index,
                        key=f"sidebar_q_{i}",
                    )
                    st.session_state.sidebar_answers[i] = answer


if __name__ == "__main__":
    main()
