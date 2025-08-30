# Clinical Myocardial Infarction (MI) Abstraction Pipeline

This project implements a sophisticated pipeline to automate the clinical abstraction of Myocardial Infarction (MI) and its onset date from electronic health records, based on the MIMIC-IV demo dataset.

It uses a combination of rule-based logic and optional Large Language Model (LLM) powered analysis of clinical notes to identify evidence, evaluate it against clinical guidelines, and produce a structured, requirement-compliant output.

## 🚀 **NEW: Patient-Level Historical Analysis**

The pipeline now supports **comprehensive patient-level analysis** that goes beyond single hospital admissions to analyze complete patient visit histories:

*   **Historical Evidence Aggregation**: Collects and analyzes evidence across all patient visits chronologically
*   **Enhanced Onset Date Resolution**: Uses complete patient timeline for more accurate MI onset date determination
*   **Comprehensive Debugging**: Advanced logging system with detailed step-by-step analysis tracking
*   **Scalable Architecture**: Designed for processing thousands of patients with efficient caching and batch processing

## Key Features

### Core Analysis Capabilities
*   **Modular Pipeline Design**: The system is broken down into clear, reusable components for data loading, evidence collection, rule evaluation, and onset date resolution.
*   **Dual Analysis Modes**: 
    *   **Admission-Level**: Traditional single-visit analysis for real-time diagnosis
    *   **Patient-Level**: Comprehensive historical analysis across all patient visits
*   **Guideline-Compliant Logic**: Implements the official clinical abstraction guidelines for MI, including a hierarchical approach for determining the MI onset date.

### Advanced Features
*   **LLM-Powered Evidence Extraction**: Optionally uses an LLM (via Together.ai) to perform nuanced extraction of clinical evidence from unstructured discharge notes.
*   **Historical Timeline Analysis**: LLM-based temporal pattern analysis across complete patient visit history
*   **Resilient LLM Client**: The LLM client is built for production use, featuring:
    *   **Persistent Caching**: Saves successful LLM responses to disk (`output/llm_cache.json`) to avoid re-processing data and save costs on subsequent runs.
    *   **Robust Retries**: Automatically retries failed API requests with exponential backoff and jitter to handle temporary server issues (e.g., 503 errors).
    *   **Rate Limiting**: Prevents the client from exceeding API rate limits.

### Production-Ready Features
*   **Comprehensive Logging**: Enhanced debug logging system with structured prefixes for easy troubleshooting and monitoring
*   **Evidence Deduplication**: Intelligent aggregation and deduplication of symptoms and diagnoses across visits
*   **Performance Optimization**: Efficient data loading, caching, and batch processing capabilities
*   **Flexible Output Formats**: Generates multiple output files, including detailed per-admission JSON, a combined summary CSV, and the final, nested `requirement_compliant_output.json`.

## Data Requirements

This pipeline is designed to work with the following three CSV files from the **MIMIC-IV Demo Dataset**:

1.  `labevents.csv`
2.  `d_labitems.csv`
3.  `discharge_notes_demo.csv`

Before running the project, please ensure these three files are placed in the root directory.

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd lighten
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up LLM API Key (Optional)**:
    If you wish to use the LLM for evidence extraction, you must provide an API key.
    *   Create a file named `.env` in the project's root directory.
    *   Add your Together.ai API key to the file like this:
        ```
        TOGETHER_API_KEY="your_api_key_here"
        ```
    *   If no API key is provided, the pipeline will still run but will skip the LLM-based steps.

## How to Run

### Admission-Level Analysis (Original)

To run the traditional admission-level pipeline and process all available patient admissions:

```bash
python run.py --all
```

The script will process all admissions that have both lab data and clinical notes, displaying a progress bar and detailed logs in your terminal.

**Targeted Processing:**
*   **Process a specific patient**: `python run.py --patient-id <patient_id>`
*   **Process a specific admission**: `python run.py --hadm-id <hadm_id>`

### 🆕 Patient-Level Historical Analysis

To run the new patient-level analysis that processes complete patient visit histories:

```bash
# Demo analysis for a single patient
python run_patient_level_analysis.py --demo --patient-id 12345

# Process all patients in the dataset
python run_patient_level_analysis.py --full-run

# Enable debug logging for detailed analysis tracking
python run_patient_level_analysis.py --demo --patient-id 12345 --log-level DEBUG
```

### Validation and Testing

Validate the patient-level pipeline functionality:

```bash
# Run comprehensive validation tests
python validate_patient_level_pipeline.py --discharge-notes discharge_notes_demo.csv --lab-events labevents.csv --lab-items d_labitems.csv

# Run validation with debug logging
python validate_patient_level_pipeline.py --log-level DEBUG --output validation_results.json
```

## Output Files

### Admission-Level Analysis Output
All admission-level output is saved to the `output/` directory:

*   `requirement_compliant_output.json`: The primary, final output file. It contains a nested JSON object keyed by patient ID, with the MI status and onset date, formatted exactly as specified in the project requirements.
*   `combined_results.csv`: A flattened CSV summary of all processed admissions, useful for quick analysis.
*   `combined_results.json`: A JSON file containing the detailed results for all processed admissions, keyed by `hadm_id`.
*   `admission_{hadm_id}_results.json`: A detailed JSON file is saved for each individual admission that is processed.
*   `llm_cache.json`: A file containing the cached responses from the LLM API. This file is used to avoid redundant API calls on subsequent runs.

### 🆕 Patient-Level Analysis Output
Patient-level analysis generates comprehensive historical analysis results:

*   `patient_level_results.json`: Complete patient-level analysis results with historical evidence aggregation
*   `patient_level_summary.csv`: Summary CSV with patient-level MI diagnosis and onset dates
*   `patient_{patient_id}_analysis.json`: Detailed analysis for individual patients including:
    *   Complete visit timeline and evidence aggregation
    *   MI diagnosis with supporting evidence
    *   Historical onset date resolution with reasoning
    *   Metadata including visit counts, date ranges, and processing statistics

## Project Structure

```
lighten/
├── lighten_ml/
│   ├── data_loaders/     # Modules for loading and preprocessing data
│   │   ├── patient_history_loader.py  # 🆕 Patient visit history management
│   │   └── lab_data_loader.py         # Enhanced with patient-level methods
│   ├── evidence_collectors/ # Modules for extracting evidence from data
│   ├── pipeline/         # The main clinical pipeline orchestration
│   │   ├── clinical_pipeline.py       # Original admission-level pipeline
│   │   └── patient_level_pipeline.py  # 🆕 Patient-level historical analysis
│   ├── resolvers/        # Logic for resolving the MI onset date
│   │   └── onset_date_resolver.py     # Enhanced with historical analysis
│   └── rule_engines/     # The core MI detection rule engine
├── examples/             # 🆕 Usage examples and integration guides
│   └── patient_level_examples.py      # Comprehensive usage examples
├── output/               # Default directory for all output files
├── tests/                # Unit and integration tests
├── .env.example          # Example environment file
├── run.py                # Original admission-level analysis script
├── run_patient_level_analysis.py      # 🆕 Patient-level analysis script
├── validate_patient_level_pipeline.py # 🆕 Validation and testing script
├── PATIENT_LEVEL_ANALYSIS.md          # 🆕 Comprehensive documentation
├── INTEGRATION_GUIDE.md               # 🆕 Integration and deployment guide
└── requirements.txt      # Project dependencies

## 📚 Documentation

### Core Documentation
- **README.md**: This file - overview and quick start guide
- **PATIENT_LEVEL_ANALYSIS.md**: Comprehensive patient-level analysis documentation
- **INTEGRATION_GUIDE.md**: Production integration and deployment guide

### Examples and Validation
- **examples/patient_level_examples.py**: Complete usage examples for all scenarios
- **validate_patient_level_pipeline.py**: Comprehensive validation and testing suite

## 🔍 Analysis Comparison

| Feature | Admission-Level | Patient-Level |
|---------|----------------|---------------|
| **Scope** | Single hospital visit | Complete patient history |
| **Evidence Collection** | One admission's data | Aggregated across all visits |
| **Onset Date Resolution** | Visit-specific analysis | Historical timeline analysis |
| **Accuracy** | Limited by single visit context | Enhanced by complete patient context |
| **Use Case** | Real-time diagnosis | Comprehensive retrospective analysis |
| **Processing Time** | Fast (seconds) | Moderate (depends on visit count) |
| **Memory Usage** | Low | Higher (proportional to patient history) |

## 🚀 Getting Started

### Quick Start - Admission Level
```bash
# Install dependencies
pip install -r requirements.txt

# Run admission-level analysis
python run.py --all
```

### Quick Start - Patient Level
```bash
# Run patient-level demo
python run_patient_level_analysis.py --demo --patient-id 12345

# Validate installation
python validate_patient_level_pipeline.py --log-level INFO
```
