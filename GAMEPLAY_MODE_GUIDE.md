# Gameplay Mode Guide

## Overview

The **Gameplay Mode** is an optional feature that allows you to use template-based configurations for datasets containing exercises with different gameplay types. Instead of manually selecting and describing columns for each analysis session, you can create a single gameplay configuration JSON file that automatically populates all settings based on the selected gameplay type.

## When to Use Gameplay Mode

Use Gameplay Mode when:
- Your dataset contains exercises with different gameplay types (e.g., multiple choice, fill-in-the-blank, drag-and-drop)
- Each gameplay type has specific columns and requires different annotation instructions
- You want to analyze one gameplay type at a time
- Your dataset has a `gameplay_athena` column that identifies the gameplay type for each row

## How It Works

### Standard Mode vs. Gameplay Mode

**Standard Mode** (default):
1. Upload your dataset
2. Manually select columns to analyze
3. Manually provide column descriptions
4. Manually enter codebook and annotation instructions
5. Continue with analysis

**Gameplay Mode**:
1. Upload your dataset
2. Check "Use gameplay template configuration?"
3. Upload your gameplay configuration JSON
4. Select gameplay type from dropdown
5. All settings are automatically loaded
6. Continue directly to model selection (Step 5)

## Configuration File Structure

A gameplay configuration JSON file must contain:

1. **`common_config`**: Settings shared across all gameplays
2. **`common_columns_to_all_gp`**: Column descriptions for columns present in all gameplays
3. **`gameplays`**: Specific configurations for each gameplay type

### Example Structure

```json
{
  "common_config": {
    "codebook": "General annotation instructions...",
    "examples": "Optional examples...",
    "selected_fields": ["Reasoning", "Classification"],
    "label_column": "Classification",
    "label_type": "Integer",
    "annotation_columns": ["Rater_1"],
    "text_columns": ["instruction", "module_description"]
  },
  "common_columns_to_all_gp": {
    "module_id": "Module identifier",
    "module_title": "Module title",
    "instruction": "General instruction for the student"
  },
  "gameplays": {
    "input_line_global": {
      "description": "Fill-in-the-blank exercise",
      "columns": {
        "line": "Text with blanks to fill",
        "FBI0": "Feedback for wrong answer"
      },
      "codebook": "Specific instructions for fill-in-the-blank...",
      "text_columns": ["line", "FBI0"]
    },
    "MSM_qcu": {
      "description": "Multiple choice with single correct answer",
      "columns": {
        "proposition1": "First answer option",
        "proposition2": "Second answer option",
        "proposition3": "Third answer option",
        "proposition4": "Fourth answer option",
        "correct_answer": "Number of correct option",
        "FBI0": "Feedback for wrong answer"
      },
      "codebook": "Specific instructions for MCQ...",
      "text_columns": ["proposition1", "proposition2", "proposition3", "proposition4", "FBI0"]
    }
  }
}
```

## Field Descriptions

### Common Config Fields

- **`codebook`**: Base annotation instructions used across all gameplays (can be overridden per gameplay)
- **`examples`**: Optional examples to guide the LLM
- **`selected_fields`**: Fields the LLM should extract (e.g., ["Reasoning", "Classification"])
- **`label_column`**: Which field contains the main classification label
- **`label_type`**: Data type of labels ("Integer", "Float", or "Text")
- **`annotation_columns`**: Columns containing human annotations for comparison
- **`text_columns`**: Common columns that should be normalized/cleaned

### Common Columns

Dictionary mapping column names to their descriptions. These columns are expected to be present in all gameplay types.

### Gameplay-Specific Fields

For each gameplay type, you can specify:

- **`description`**: Human-readable description of the gameplay type
- **`columns`**: Dictionary of gameplay-specific column names and descriptions
- **`codebook`**: Gameplay-specific annotation instructions (overrides common codebook)
- **`examples`**: Gameplay-specific examples (optional, overrides common examples)
- **`text_columns`**: Additional text columns specific to this gameplay

## Merging Logic

When you select a gameplay, the system automatically:

1. **Merges column descriptions**: Combines `common_columns_to_all_gp` with the gameplay's `columns`
2. **Merges text columns**: Combines common `text_columns` with gameplay-specific ones
3. **Selects codebook**: Uses gameplay-specific `codebook` if present, otherwise uses common one
4. **Selects examples**: Uses gameplay-specific `examples` if present, otherwise uses common ones
5. **Auto-selects all columns**: Automatically includes all merged columns in the analysis

## Step-by-Step Usage

### 1. Prepare Your Dataset

Ensure your dataset (CSV or Excel) contains:
- A `gameplay_athena` column identifying the gameplay type for each row
- All columns mentioned in your configuration file
- Human annotations in the specified annotation columns

### 2. Create Your Configuration File

Create a JSON file following the structure above. You can use `data/sample_gameplay_config.json` as a template.

### 3. Upload and Configure in Streamlit

1. **Step 1: Upload Dataset**
   - Upload your CSV or Excel file
   - Check "Use gameplay template configuration?"
   - Upload your gameplay configuration JSON
   - Select the gameplay type from the dropdown
   - Preview shows what will be auto-loaded

2. **Step 2: Data Selection**
   - This step is automated in gameplay mode
   - Configuration is loaded from the template
   - Data is automatically filtered to the selected gameplay
   - Text columns are automatically normalized

3. **Step 3-4**: Codebook and fields are pre-loaded (read-only)

4. **Step 5**: Continue with model selection as usual

## Validation

The system automatically validates:

- ✅ Configuration contains `"gameplays"` section
- ✅ Selected gameplay exists in `gameplay_athena` column
- ✅ All required columns exist in the dataset
- ✅ No missing or malformed data

If validation fails, clear error messages indicate what needs to be fixed.

## Example Workflow

```
Dataset: exercises.xlsx (1000 rows)
- 300 rows: gameplay_athena = "input_line_global"
- 400 rows: gameplay_athena = "MSM_qcu"
- 300 rows: gameplay_athena = "drag_and_drop"

You want to analyze "MSM_qcu" exercises:

1. Upload exercises.xlsx
2. Enable gameplay mode
3. Upload gameplay_config.json
4. Select "MSM_qcu" from dropdown
   → System filters to 400 rows
   → Loads MCQ-specific columns and codebook
5. Continue with analysis on those 400 rows
```

## Saving Sessions

When using gameplay mode, your session file will include:
- `"use_gameplay_mode": true`
- `"selected_gameplay": "your_gameplay_type"`

This allows you to reload the exact same configuration in future sessions.

## Backward Compatibility

- Old configuration files (without `"gameplays"`) continue to work in standard mode
- You can switch between standard and gameplay mode as needed
- The checkbox defaults to off, so existing workflows are unchanged

## Tips and Best Practices

1. **Organize by structure**: Group gameplays that share similar column structures
2. **Reuse codebooks**: Use common codebook for general criteria, override only when gameplay needs specific instructions
3. **Document descriptions**: Write clear column descriptions—they help the LLM understand the data
4. **Test incrementally**: Start with one gameplay type, validate it works, then add others
5. **Version control**: Keep your gameplay config files in version control with your datasets

## Troubleshooting

**Error: "Column 'gameplay_athena' not found"**
- Your dataset must have a `gameplay_athena` column for gameplay mode
- Solution: Add this column or use standard mode

**Error: "Gameplay 'X' not found in dataset"**
- The selected gameplay doesn't exist in your data
- Solution: Check the available gameplays listed in the error message

**Error: "Missing required columns: [...]"**
- Your dataset is missing columns defined in the configuration
- Solution: Verify column names match exactly (case-sensitive)

**Configuration not loading**
- Ensure JSON is valid (use a JSON validator)
- Check that `"gameplays"` key exists at the top level
- Verify all required fields are present

## Sample Configuration

See `data/sample_gameplay_config.json` for a complete working example with multiple gameplay types (fill-in-the-blank, single choice MCQ, multiple choice MCQ, drag-and-drop).

## Support

For issues or questions:
1. Check this guide thoroughly
2. Verify your JSON configuration is valid
3. Test with the sample configuration first
4. Review error messages carefully—they indicate exactly what's wrong
