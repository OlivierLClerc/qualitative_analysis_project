import streamlit as st
import pandas as pd
import json
from transformers import pipeline

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


# Cache the transformer so that it is only loaded once.
@st.cache_resource(show_spinner=False)
def get_translator():
    return pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")


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

    # --- Step 1: File upload for data ---
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"], key="data_file")
    if uploaded_file is not None:
        if st.session_state.df is None:
            st.session_state.df = pd.read_csv(
                uploaded_file, delimiter=";", quoting=1
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

        # Only create the new column when the user confirms the annotator name.
        if st.session_state.new_col_name == "":
            if st.button("Confirm Annotator Name"):
                if st.session_state.annotator_name:
                    candidate_col = f"Rater_{st.session_state.annotator_name}"
                    if candidate_col not in df.columns:
                        df[candidate_col] = pd.NA
                    st.session_state.new_col_name = candidate_col
                    st.success(f"Column '{candidate_col}' created!")
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

            # --- Step 4: Select columns to display ---
            possible_columns = [
                c for c in df.columns if c != st.session_state.new_col_name
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

                rating_val = df.at[idx, st.session_state.new_col_name]
                st.markdown(f"**Current Index: {idx}**")
                st.markdown(f"**Existing Rating:** {rating_val}")
                for col in selected_columns:
                    st.write(f"**{col}:** {df.at[idx, col]}")

                # --- Row Translation Option: Translate current row on demand ---
                translate_row = st.checkbox(
                    "Translate current row to English", key="translate_row"
                )
                if translate_row:
                    # Check if this row has already been translated
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

                # --- Navigation buttons below the data ---
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Previous"):
                        save_current_answers(idx, st.session_state.sidebar_answers)
                        st.session_state.current_index = max(0, idx - 1)
                        st.rerun()
                with col2:
                    if st.button("Next"):
                        save_current_answers(idx, st.session_state.sidebar_answers)
                        st.session_state.current_index = min(len(df) - 1, idx + 1)
                        st.rerun()

            # --- Step 6: Download button ---
            st.write("**When done, you can download the updated CSV:**")
            csv_data = st.session_state.df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="Download Annotations",
                data=csv_data,
                file_name="annotated_data.csv",
                mime="text/csv",
            )

            # --- Step 7: Annotation interface in the sidebar ---
            st.sidebar.header("Annotation Questions")
            if "sidebar_answers" not in st.session_state:
                st.session_state.sidebar_answers = {}

            if st.session_state.questions:
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


def save_current_answers(row_idx, answers_dict, warn_if_incomplete=False):
    """
    Saves the user's Yes/No answers for the given row into the DataFrame.
    This function always recalculates the rating based on the current answers.
    """
    df = st.session_state.df
    new_col = st.session_state.new_col_name

    if not st.session_state.questions:
        return True

    all_filled = all(a in ("yes", "no") for a in answers_dict.values())
    if not all_filled:
        if warn_if_incomplete:
            st.warning("Not all questions answered. Row not fully saved.")
        for q_idx, val in answers_dict.items():
            st.session_state.answers[(row_idx, q_idx)] = val
        return False

    all_yes = all(a == "yes" for a in answers_dict.values())
    df.at[row_idx, new_col] = 1 if all_yes else 0

    for q_idx, val in answers_dict.items():
        st.session_state.answers[(row_idx, q_idx)] = val

    return True


if __name__ == "__main__":
    main()
