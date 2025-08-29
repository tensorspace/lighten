# Clinical Myocardial Infarction (MI) Abstraction Pipeline

This project implements a sophisticated pipeline to automate the clinical abstraction of Myocardial Infarction (MI) and its onset date from electronic health records, based on the MIMIC-IV demo dataset.

It uses a combination of rule-based logic and optional Large Language Model (LLM) powered analysis of clinical notes to identify evidence, evaluate it against clinical guidelines, and produce a structured, requirement-compliant output.

## Key Features

*   **Modular Pipeline Design**: The system is broken down into clear, reusable components for data loading, evidence collection, rule evaluation, and onset date resolution.
*   **Guideline-Compliant Logic**: Implements the official clinical abstraction guidelines for MI, including a hierarchical approach for determining the MI onset date.
*   **LLM-Powered Evidence Extraction**: Optionally uses an LLM (via Together.ai) to perform nuanced extraction of clinical evidence from unstructured discharge notes.
*   **Resilient LLM Client**: The LLM client is built for production use, featuring:
    *   **Persistent Caching**: Saves successful LLM responses to disk (`output/llm_cache.json`) to avoid re-processing data and save costs on subsequent runs.
    *   **Robust Retries**: Automatically retries failed API requests with exponential backoff and jitter to handle temporary server issues (e.g., 503 errors).
    *   **Rate Limiting**: Prevents the client from exceeding API rate limits.
*   **Comprehensive Logging**: Provides detailed, structured logs for monitoring pipeline progress, data loading, and LLM interactions, making debugging easy.
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

To run the pipeline and process all available patient admissions, use the `--all` flag:

```bash
python run.py --all
```

The script will process all admissions that have both lab data and clinical notes, displaying a progress bar and detailed logs in your terminal.

### Targeted Processing

You can also process a single patient or a single admission using the following arguments:

*   **Process a specific patient**: `python run.py --patient-id <patient_id>`
*   **Process a specific admission**: `python run.py --hadm-id <hadm_id>`

## Output Files

All output is saved to the `output/` directory:

*   `requirement_compliant_output.json`: The primary, final output file. It contains a nested JSON object keyed by patient ID, with the MI status and onset date, formatted exactly as specified in the project requirements.
*   `combined_results.csv`: A flattened CSV summary of all processed admissions, useful for quick analysis.
*   `combined_results.json`: A JSON file containing the detailed results for all processed admissions, keyed by `hadm_id`.
*   `admission_{hadm_id}_results.json`: A detailed JSON file is saved for each individual admission that is processed.
*   `llm_cache.json`: A file containing the cached responses from the LLM API. This file is used to avoid redundant API calls on subsequent runs.

## Project Structure

```
lighten/
├── lighten_ml/
│   ├── data_loaders/     # Modules for loading and preprocessing data.
│   ├── evidence_collectors/ # Modules for extracting evidence from data.
│   ├── pipeline/         # The main clinical pipeline orchestration.
│   ├── resolvers/        # Logic for resolving the MI onset date.
│   └── rule_engines/     # The core MI detection rule engine.
├── output/               # Default directory for all output files.
├── tests/                # Unit and integration tests.
├── .env.example          # Example environment file.
├── run.py                # The main script to execute the pipeline.
└── requirements.txt      # Project dependencies.
