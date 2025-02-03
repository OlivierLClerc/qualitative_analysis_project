"""
annotations.py

This module provides a GUI-based annotation tool for rating data in a CSV file.

Dependencies:
    - tkinter
    - pandas

Classes:
    - DataRaterApp: A Tkinter-based application class for loading data from a CSV, 
      displaying entries, collecting user ratings, and saving annotations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import pandas as pd

# Import your custom data-processing utilities (the load_data function)
from qualitative_analysis import load_data


class DataRaterApp:
    """
    A Tkinter-based application for interactively annotating rows in a CSV file.

    This class allows users to:
        - Browse to a CSV file
        - Specify an annotator name
        - Select columns to display
        - Answer Yes/No questions for each entry
        - Automatically record annotations (e.g., 1 if all answers are "Yes")
        - Navigate through entries (Next/Previous)
        - Optionally save the resulting DataFrame to a new CSV upon exit.

    Attributes:
    -----------
    root : tk.Tk
        The main Tkinter window.
    df : pd.DataFrame or None
        The DataFrame loaded from the CSV file.
    current_index : int
        The index of the current row being displayed/annotated.
    questions : list of str
        A list of questions to ask for each entry.
    answers : dict of int to tk.StringVar
        Maps each question index to a Tkinter StringVar storing "yes" or "no".
    new_col_name : str or None
        The name of the new rating column (e.g., "Rater_John") added to df.
    selected_columns : list of str
        Columns chosen by the user for display.

    Example:
    -------
    # Minimal usage:
    >>> import tkinter as tk
    >>> from qualitative_analysis.annotations import DataRaterApp
    >>> root = tk.Tk()
    >>> app = DataRaterApp(root)
    >>> root.mainloop()
    """

    def __init__(self, root: tk.Tk):
        """
        Initializes the DataRaterApp with a Tkinter root, sets up widgets,
        and waits for the user to load a CSV file.
        """
        self.root = root
        self.root.title("Data Rater Tool")
        self.root.geometry("800x600")

        # Intercept close-window event and run self.on_close instead
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.df: pd.DataFrame | None = None
        self.current_index: int = 0
        self.questions: list[str] = []
        self.answers: dict[int, tk.StringVar] = {}
        self.new_col_name: str | None = None
        self.selected_columns: list[str] = []

        # Initial UI: just a button to load the data
        self.load_button = ttk.Button(
            self.root, text="Load Data File", command=self.browse_file
        )
        self.load_button.pack(pady=20)

        # Prepare the rest of frames, but don't pack them yet
        self.setup_ui()

    def browse_file(self):
        """
        Prompts the user to select a CSV file via file dialog, then loads it into self.df.
        Asks for an annotator name, creates a rating column for that annotator,
        and reveals the main interface.
        """
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )

        if not file_path:
            messagebox.showwarning("No File Selected", "No CSV file was chosen.")
            return  # Do nothing, user can try again

        try:
            # Use your custom load_data function
            self.df = load_data(
                file=file_path, file_type="csv", delimiter=";", quoting=1
            )
            self.df = self.df.reset_index(drop=True)

            # Prompt for annotator name
            annotator_name = simpledialog.askstring(
                "Annotator Name", "Please enter your name:"
            )
            if not annotator_name:
                annotator_name = "Unknown"

            self.new_col_name = f"Rater_{annotator_name}"

            if self.new_col_name not in self.df.columns:
                # Initialize with missing values so that un-coded rows stay blank
                self.df[self.new_col_name] = pd.NA

            # Hide the load_button now that file is loaded
            self.load_button.pack_forget()

            # Show the rest of the UI
            self.show_main_interface()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the file: {e}")

    def setup_ui(self):
        """
        Creates (but doesn't pack) frames and widgets for:
        - Question input
        - Column selection
        - Data display
        - Navigation
        - Q&A area
        """
        # Frame for adding questions
        self.question_frame = ttk.LabelFrame(self.root, text="Input Questions")
        self.question_entry = ttk.Entry(self.question_frame, width=50)
        self.add_q_button = ttk.Button(
            self.question_frame, text="Add Question", command=self.add_question
        )

        # Frame to let users select which columns to view
        self.column_frame = ttk.LabelFrame(
            self.root, text="Select Columns (Ctrl+Click to select multiple)"
        )
        self.column_listbox = tk.Listbox(
            self.column_frame, selectmode=tk.MULTIPLE, exportselection=False, height=8
        )
        self.scrollbar = ttk.Scrollbar(
            self.column_frame, orient="vertical", command=self.column_listbox.yview
        )
        self.column_listbox.configure(yscrollcommand=self.scrollbar.set)

        self.confirm_button = ttk.Button(
            self.column_frame, text="Confirm Selected Columns", command=self.set_columns
        )

        # Frame to display the current data row
        self.data_frame = ttk.LabelFrame(self.root, text="Current Entry")
        self.text_display = tk.Text(
            self.data_frame, wrap="word", height=10, font=("Arial", 10)
        )

        # Navigation (previous/next) buttons
        self.nav_frame = ttk.Frame(self.root)
        self.prev_button = ttk.Button(
            self.nav_frame, text="Previous", command=self.prev_entry
        )
        self.next_button = ttk.Button(
            self.nav_frame, text="Next", command=self.next_entry
        )

        # Question answering area
        self.qa_frame = ttk.LabelFrame(self.root, text="Answer Questions")

    def show_main_interface(self):
        """
        Packs (displays) the frames and widgets that require a loaded DataFrame,
        then calls start_evaluation to begin annotation.
        """
        self.question_frame.pack(pady=10, padx=10, fill="x")
        self.question_entry.pack(side="left", padx=5)
        self.add_q_button.pack(side="left", padx=5)

        self.column_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.column_listbox.pack(side="left", fill="both", expand=True, padx=5)
        self.scrollbar.pack(side="right", fill="y")
        self.confirm_button.pack(pady=5)

        # Populate the listbox with columns EXCEPT the rating column
        assert self.df is not None
        for col in self.df.columns:
            if col != self.new_col_name:
                self.column_listbox.insert(tk.END, col)

    def add_question(self):
        """Adds the user's question from question_entry into self.questions."""
        question = self.question_entry.get()
        if question:
            self.questions.append(question)
            self.question_entry.delete(0, tk.END)
            messagebox.showinfo("Success", f"Added question: {question}")

    def set_columns(self):
        """
        Collects the user's selected columns for display
        and starts the evaluation process if any are selected.
        """
        selected_indices = self.column_listbox.curselection()
        self.selected_columns = [self.column_listbox.get(i) for i in selected_indices]

        if self.selected_columns:
            messagebox.showinfo(
                "Success", f"Selected columns: {', '.join(self.selected_columns)}"
            )
            self.start_evaluation()
        else:
            messagebox.showwarning("Warning", "Please select at least one column")

    def start_evaluation(self):
        """
        Hides the question/column selection frames,
        figures out where to start coding (e.g. first NA row),
        and shows the data display area, navigation, and Q&A frame.
        """
        self.question_frame.pack_forget()
        self.column_frame.pack_forget()

        self.data_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.text_display.pack(padx=5, pady=5, fill="both", expand=True)

        self.nav_frame.pack(pady=10)
        self.prev_button.pack(side="left", padx=5)
        self.next_button.pack(side="left", padx=5)

        self.qa_frame.pack(pady=10, padx=10, fill="x")

        # If there's at least one NA row in Rater_..., start there
        assert self.df is not None
        na_rows = self.df[self.df[self.new_col_name].isna()].index
        if len(na_rows) > 0:
            self.current_index = na_rows[0]
        else:
            self.current_index = 0  # If everything is coded, start at 0

        self.load_entry()

    def load_entry(self):
        """
        Loads and displays the current row (self.current_index),
        creates radio-button questions, and stores user responses in self.answers.
        """
        self.text_display.delete("1.0", tk.END)
        for widget in self.qa_frame.winfo_children():
            widget.destroy()

        # Show rating column first (so user knows if row is coded)
        assert self.df is not None
        rating_value = self.df.at[self.current_index, self.new_col_name]
        self.text_display.insert(
            tk.END, f"=== {self.new_col_name} ===\n{rating_value}\n\n"
        )

        # Then show selected columns
        for col in self.selected_columns:
            entry_text = f"=== {col} ===\n{self.df.at[self.current_index, col]}\n\n"
            self.text_display.insert(tk.END, entry_text)

        # Create radio buttons for each question
        self.answers = {}
        for i, question in enumerate(self.questions):
            frame = ttk.Frame(self.qa_frame)
            frame.pack(fill="x", pady=2)

            ttk.Label(frame, text=question).pack(side="left", padx=5)
            var = tk.StringVar(value="")
            ttk.Radiobutton(frame, text="Yes", variable=var, value="yes").pack(
                side="left"
            )
            ttk.Radiobutton(frame, text="No", variable=var, value="no").pack(
                side="left"
            )
            self.answers[i] = var

    def row_is_already_coded(self) -> bool:
        """
        Returns True if the row at self.current_index is already rated
        (i.e. not NA in self.new_col_name).
        """
        assert self.df is not None
        val = self.df.at[self.current_index, self.new_col_name]
        return pd.notna(val)

    def check_or_confirm_incomplete(self) -> bool:
        """
        If row is not coded and user hasn't answered all questions,
        pop up a "Are you sure?" box. Return True if user wants to proceed,
        False otherwise.
        """
        # If the row is already coded, no need to recheck
        if self.row_is_already_coded():
            return True

        # If row not coded, see if user has answered all questions
        all_filled = all(var.get() for var in self.answers.values())
        if all_filled:
            # Then we can save it right away
            return self.save_current_answers(force_save=True)
        else:
            # It's incomplete: ask user if they want to skip
            skip = messagebox.askyesno(
                "Incomplete Answers",
                "You did not answer all questions. Proceed without saving?",
            )
            return skip  # If True, user doesn't want to fill them => skip & move

    def save_current_answers(self, force_save: bool = False) -> bool:
        """
        If the user filled everything, save answers to self.df.
        If not all filled and force_save=False, show error and return False.
        If not all filled and force_save=True, still do the same check.

        Returns True if the row was saved or if the row was already coded,
        or if user doesn't care to fill them. Otherwise False if blocked.
        """
        if not self.questions:
            # No questions => nothing to save
            return True

        # If row is already coded, we do not require re-answering
        if self.row_is_already_coded() and not force_save:
            return True

        # Check if all are answered
        all_filled = all(var.get() for var in self.answers.values())
        if not all_filled:
            if not force_save:
                # Normal path: show error
                messagebox.showerror(
                    "Error", "Please answer all questions before moving on!"
                )
                return False
            else:
                # force_save path: user explicitly said "skip answers, continue"
                return True

        # If all answered, set rating to 1 if all 'yes', otherwise 0
        all_yes = all(var.get() == "yes" for var in self.answers.values())
        assert self.df is not None, "df should be loaded by now."
        self.df.at[self.current_index, self.new_col_name] = 1 if all_yes else 0
        return True

    def prev_entry(self):
        """Navigates to the previous row, checking if we can move away safely."""
        if self.current_index > 0:
            # 1) Prompt or forcibly check if row is incomplete
            if self.check_or_confirm_incomplete():
                # 2) If user is okay to proceed
                self.current_index -= 1
                self.load_entry()

    def next_entry(self):
        """Navigates to the next row, checking if we can move away safely."""
        if self.current_index < len(self.df) - 1:
            if self.check_or_confirm_incomplete():
                self.current_index += 1
                self.load_entry()

    def on_close(self):
        """
        Called when the user tries to close the application window.

        - If row is incomplete, ask user if they want to skip or fill them out.
        - If data is loaded, optionally save the entire DataFrame to CSV.
        - Finally, destroy the window.
        """
        if self.df is not None:
            if not self.check_or_confirm_incomplete():
                # user canceled
                return

            # Ask if we should save the entire DataFrame to CSV
            answer = messagebox.askyesno(
                "Save File", "Would you like to save your annotations before exiting?"
            )
            if answer:
                save_path = filedialog.asksaveasfilename(
                    title="Save CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                )
                if save_path:
                    try:
                        self.df.to_csv(save_path, index=False, encoding="utf-8-sig")
                        messagebox.showinfo("Saved", f"Data saved to {save_path}")
                    except Exception as e:
                        messagebox.showerror("Save Error", f"Failed to save: {e}")

        self.root.destroy()
