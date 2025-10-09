# Generation Mode Guide

## Overview

The new **Generation Mode** in LLM4Humanities allows you to generate new content using LLMs based on blueprint examples, then automatically annotate the generated content. This is perfect for creating exercises, questions, or any structured content variations.

## Workflow Steps

### 1. Blueprint Input
- **Purpose**: Provide reference examples to guide content generation
- **Options**: 
  - Text input (multiple blueprints supported)
  - File upload (.txt files with optional delimiter splitting)
- **Example Use Cases**:
  - Exercise generation with sample exercises
  - Question generation with template questions
  - Content variation creation

### 2. Generation Configuration
- **Generation Prompt**: Describe desired variations (e.g., "Generate similar exercises with different numerical values and varying difficulty levels")
- **Number of Items**: Specify how many items to generate (1-100)
- **Output Columns**: Define the structure of generated content
  - Column names and descriptions
  - Example: `Exercise_Content`, `Expected_Answer`, `Option_A`, `Option_B`, etc.
- **Advanced Parameters**: Temperature (creativity) and max tokens

### 3. LLM Configuration
- **Reuses existing LLM setup**: Provider selection, API key, model choice
- **Supported Providers**: OpenAI, Anthropic, Gemini, Together, OpenRouter, Azure

### 4. Content Generation
- **Cost Estimation**: Preview costs before generation
- **Progress Tracking**: Real-time progress with success/failure counts
- **Debug Mode**: View generated prompts for troubleshooting
- **Results**: Preview generated content in tabular format
- **Download**: Export generated content as Excel file

### 5. Annotation Configuration
- **Annotation Prompt**: Define how to evaluate generated content (e.g., "Verify similarity with blueprint, validate answer correctness, assess difficulty")
- **Annotation Columns**: Specify evaluation fields
  - Example: `Similarity_Score`, `Answer_Validity`, `Difficulty_Level`, `Comments`
- **Advanced Parameters**: Lower temperature recommended for consistent annotations

### 6. Content Annotation
- **Automated Evaluation**: LLM annotates each generated item
- **Cost Estimation**: Preview annotation costs
- **Progress Tracking**: Real-time annotation progress
- **Results**: Combined dataset with both generated content and annotations
- **Multi-Sheet Export**: 
  - Annotated Content (complete dataset)
  - Original Generated (content only)
  - Annotations Only (evaluation fields)

## Key Features

### ✅ Advantages over Annotation Mode
- **No human annotations required**: Perfect for rapid content creation
- **Scalable content generation**: Create hundreds of variations quickly
- **Quality control**: Built-in LLM-based quality assessment
- **Flexible structure**: Define custom output formats
- **Cost-effective**: Generate training data or content variations efficiently

### 🔧 Reused Components
- **LLM Configuration**: Same provider/model selection as Annotation Mode
- **Session Management**: Save/load generation configurations
- **Error Handling**: Robust parsing and fallback mechanisms
- **Progress Tracking**: Real-time feedback during processing

### 📊 Export Options
- **Excel Downloads**: Multi-sheet workbooks with different data views
- **Separate Datasets**: Generated content and annotations as separate sheets
- **Metadata**: Generation IDs and processing information included

## Example Use Case: Exercise Generation

### Input Blueprint:
```
Calculate the area of a rectangle with length 5 cm and width 3 cm.
Answer: Area = length × width = 5 × 3 = 15 cm²
```

### Generation Prompt:
```
Generate similar geometry exercises with different shapes (rectangle, triangle, circle) and different numerical values. Vary the difficulty by using decimal numbers or larger integers.
```

### Output Columns:
- `Exercise_Question`: The problem statement
- `Expected_Answer`: Complete solution with steps
- `Difficulty_Level`: Easy/Medium/Hard classification
- `Shape_Type`: Rectangle/Triangle/Circle

### Annotation Prompt:
```
Evaluate the generated exercise for mathematical accuracy, clarity of the question, and appropriateness of difficulty level. Check that the answer matches the question.
```

### Annotation Columns:
- `Mathematical_Accuracy`: Correct/Incorrect
- `Question_Clarity`: Clear/Unclear/Ambiguous  
- `Difficulty_Rating`: 1-5 scale
- `Overall_Quality`: Pass/Fail

## Getting Started

1. **Switch to Generation Mode**: Select "Generation Mode" at the top of the app
2. **Provide Blueprints**: Add your reference examples in Step 1
3. **Configure Generation**: Define what to generate and output structure in Step 2
4. **Setup LLM**: Configure your API and model in Step 3
5. **Generate Content**: Run the generation process in Step 4
6. **Configure Annotation**: Define evaluation criteria in Step 5  
7. **Annotate Content**: Run automated evaluation in Step 6
8. **Download Results**: Export your complete dataset

## Tips for Best Results

### Blueprint Quality
- Provide clear, well-structured examples
- Include multiple blueprints if you want variety
- Ensure blueprints represent your desired output format

### Generation Prompts
- Be specific about variations you want
- Mention difficulty levels, numerical ranges, etc.
- Specify any constraints or requirements

### Column Design
- Use descriptive column names
- Provide clear descriptions for each column
- Consider your downstream use case

### Annotation Setup
- Use low temperature (0.0-0.3) for consistent evaluations
- Design evaluation criteria that are objective
- Include both quantitative and qualitative measures

This new functionality transforms LLM4Humanities from a pure analysis tool into a comprehensive content generation and evaluation platform!
