# QUALITATIVE_ANALYSIS_PROJECT

A Python-based toolkit for qualitative data analysis using language models.
The goal is to provide a tool for the automatic annotation of qualitative data through:

- A no-code mode using the interactive web interface (Streamlit app).
- A low-code mode using Jupyter notebooks for more customizable workflows.

## Installation

1. Clone the repository.
2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
3. Put your API credential.

Copy or rename `.env.example` to `.env` (if available). Populate it with your LLM credentials (e.g., Azure or Together keys, endpoints).

Example:

```bash
AZURE_API_KEY="your_azure_api_key"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_API_VERSION="2023-05-15"
```

If you’re using Together, set:

```bash
TOGETHER_API_KEY="your_together_api_key"
```

## Usage

The project offers two primary usage modes:

1. [Streamlit app](app.py): Use the interactive GUI for classification workflows.
2. [Notebooks](notebooks): Run classification workflows in Jupyter Notebooks.

### Usage 1: Streamlit app

```bash
streamlit run app.py
```
This launches a browser-based interface to upload data, configure LLMs, run analyses, and download results.

### Usage 2: Notebooks

The notebooks contain the classification workflows for each criterion:

- [Binary criterion](notebooks/notebook_binary.ipynb).
- [Multiclass criterion](notebooks/notebook_multiclass.ipynb).
- [Sequential binary criterion](notebooks/notebook_sequential_binary.ipynb).


## Project Structure

```bash
QUALITATIVE_ANALYSIS_P
├── codebook
│   ├── binary_codebook.txt
│   └── multiclass_codebook.txt
├── data
│   ├── outputs/
│   ├── multiclass_KA.csv
│   ├── multiclass_MC.csv
│   ├── multiclass_sample_chem.csv
│   ├── multiclass_sample.csv
│   └── sequential_binary_sample.csv
├── notebooks
│   ├── notebook_binary.ipynb
│   ├── notebook_multiclass.ipynb
│   └── notebook_sequential_binary.ipynb
├── qualitative_analysis
│   ├── __init__.py
│   ├── config.py
│   ├── cost_estimation.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── model_interaction.py
│   ├── notebooks_functions.py
│   ├── parsing.py
│   └── prompt_construction.py
├── .env
├── .gitignore
├── .pre-commit-config.yaml
├── app.py
├── mypy.ini
└── README.md
```
## 📂 Main Files and folders

### `app.py`
A Streamlit app providing a GUI workflow for uploading data, configuring LLMs, classifying text, and optionally comparing with external judgments.

### `notebooks/`
Contains Jupyter notebooks demonstrating how to use the library for:
- Binary classification
- Multiclass classification
- Sequential binary classification

### `data/`
Holds CSV samples (for various classification scenarios), plus an `outputs/` subfolder where processed results can be saved.

### `qualitative_analysis/`
The main Python package, housing modules

### `codebook/`
Contains human-readable text files defining classification rules or codebooks (binary/multiclass).

## 📄 Other Files

- **`.env`** – Environment variables for sensitive credentials (e.g., API keys, endpoints).
- **`.pre-commit-config.yaml`** – Config for pre-commit hooks (linting, formatting, etc.).
- **`mypy.ini`** – Configuration for static type checks (mypy).

## 🤝 Contributing

- **Coding Style**: This repo uses type hints and docstrings heavily (see `mypy.ini` for static checks).
- **Pre-Commit Hooks**: Use `.pre-commit-config.yaml` for linting/formatting. Install pre-commit hooks:

    ```bash
    pre-commit install
    ```

- **Pull Requests**: Please branch off `main` or `dev`, open a PR, and ensure you pass all lint/tests.

## 📜 License

(Include your chosen license here, e.g., MIT, Apache 2.0, etc.)

## 📚 Acknowledgments / References

- OpenAI / Azure Docs
- Together AI Documentation
- Streamlit Official Docs
- Pandas User Guide
