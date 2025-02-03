# manual_annotator.py
import tkinter as tk
from qualitative_analysis import (
    DataRaterApp,
)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataRaterApp(root)
    root.mainloop()
