# Lighten ML - Clinical Data Processing System

A system for detecting Myocardial Infarction (MI) and extracting relevant clinical variables from structured lab data and unstructured clinical notes.

## Features

- **Data Loading**: Handles large lab events and clinical notes datasets efficiently
- **Evidence Collection**: Extracts relevant clinical evidence from multiple sources
  - Troponin level analysis from lab results
  - Clinical symptoms from notes
  - ECG findings from clinical documentation
- **Rule Engine**: Implements the Universal Definition of MI criteria
- **Flexible Output**: Generates detailed JSON and CSV reports

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd lighten
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

1. Download the MIMIC-IV demo dataset from [PhysioNet](https://www.physionet.org/content/mimic-iv-demo/2.2/)
2. Place the following files in the project root directory:
   - `hosp/labevents.csv` (from the MIMIC-IV demo dataset)
   - `hosp/d_labitems.csv` (from the MIMIC-IV demo dataset)
   - `discharge_notes_demo.csv` (provided with this repository)

## Usage

### Command Line Interface

The main entry point is `run.py`, which provides a command-line interface to process patient data.

```bash
# Show help
python run.py --help

# Process a specific patient
python run.py --patient 10001

# Process a random sample of 5 patients
python run.py --sample 5

# Process all available patients
python run.py --all

# Specify custom file paths
python run.py \
    --lab-events path/to/labevents.csv \
    --lab-items path/to/d_labitems.csv \
    --clinical-notes path/to/discharge_notes_demo.csv \
    --output-dir custom_output
```

### Programmatic Usage

You can also use the pipeline programmatically:

```python
from lighten_ml.pipeline import ClinicalPipeline

# Initialize the pipeline
pipeline = ClinicalPipeline(
    lab_events_path='hosp/labevents.csv',
    lab_items_path='hosp/d_labitems.csv',
    clinical_notes_path='discharge_notes_demo.csv',
    output_dir='output'
)

# Process a single patient
result = pipeline.process_patient('10001')
print(f"MI Detected: {result['results']['mi_detected']}")
print(f"Confidence: {result['results']['confidence']:.2f}")

# Process multiple patients
patient_ids = ['10001', '10002', '10003']
results = pipeline.process_patients(patient_ids)
```

## Output

Results are saved in the specified output directory (default: `output/`):

- `patient_<ID>_results.json`: Detailed results for individual patients
- `combined_results.json`: Combined results for all processed patients
- `combined_results.csv`: Tabular summary of results

## Project Structure

```
lighten_ml/
├── data_loaders/         # Data loading and preprocessing
├── evidence_collectors/  # Evidence extraction components
├── pipeline/             # Main processing pipeline
├── rule_engines/         # Clinical decision rules
├── tests/                # Unit tests
├── run.py                # Command-line interface
└── requirements.txt      # Python dependencies
```

## Configuration

The system's behavior can be customized by modifying the configuration in the `MIRuleEngineConfig` class or by passing a configuration dictionary when initializing the pipeline.

### LLM Configuration

This system can optionally use a Large Language Model (LLM) for more advanced evidence extraction from clinical notes.

#### Enabling the LLM

To enable LLM features, you must provide an API key. The system is configured to use Together.ai by default, but can be pointed to any OpenAI-compatible API.

You can provide the API key in one of three ways:
1.  Set the `TOGETHER_API_KEY` environment variable.
2.  Set the `LLM_API_KEY` environment variable.
3.  Pass it in the pipeline configuration dictionary.

#### Configuration Options

LLM behavior can be configured through environment variables or a configuration dictionary passed to the `ClinicalPipeline`.

##### Environment Variables

-   `TOGETHER_API_KEY` or `LLM_API_KEY`: Your API key.
-   `LLM_MODEL`: The name of the model to use (e.g., `mistralai/Mixtral-8x7B-Instruct-v0.1`).
-   `LLM_BASE_URL`: The base URL for the LLM API (e.g., `https://api.together.xyz/v1`).

##### Pipeline Configuration Dictionary

You can pass a configuration dictionary when initializing the `ClinicalPipeline` to customize LLM settings.

```python
llm_config = {
    "llm": {
        "api_key": "YOUR_API_KEY",
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "base_url": "https://api.together.xyz/v1",
        "rate_limit_tps": 5,  # Requests per second
        "cache_size": 1024     # Number of LLM responses to cache
    },
    "evidence_collectors": {
        "max_notes_per_admission": 20 # Max notes to process per admission
    }
}

pipeline = ClinicalPipeline(config=llm_config, ...)
```

-   `rate_limit_tps`: Controls the number of requests per second sent to the LLM API to prevent rate-limiting errors.
-   `cache_size`: The number of LLM responses to keep in an in-memory cache to avoid redundant API calls for the same note content.
-   `max_notes_per_admission`: Limits the number of notes processed for each evidence type within a single admission to control cost and latency.

If no API key is provided, the system will gracefully fall back to using regex-based extraction methods.

## Testing

Run the test suite with:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Overview

This system processes clinical data to detect Myocardial Infarction (MI) by analyzing both structured lab results and unstructured clinical notes. It implements the Universal Definition of MI criteria, including:

- Troponin level analysis with dynamic trend detection
- Clinical symptom extraction from notes
- ECG findings analysis
- Rule-based decision making with confidence scoring

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the MIMIC-IV demo dataset** from [PhysioNet](https://www.physionet.org/content/mimic-iv-demo/2.2/) and place these files in your project root:
   - `hosp/labevents.csv`
   - `hosp/d_labitems.csv`
   - `discharge_notes_demo.csv` (provided)

3. **Run the analysis**:
   ```bash
   # Process a single patient
   python run.py --patient 10001
   
   # Or process a sample of patients
   python run.py --sample 5
   
   # Or process all patients
   python run.py --all
   ```

4. **View results** in the `output/` directory.

## System Architecture

The system is built with a modular architecture:

1. **Data Loaders**: Handle efficient loading and preprocessing of lab data and clinical notes
2. **Evidence Collectors**: Extract relevant clinical evidence from different data sources
3. **Rule Engine**: Applies clinical decision rules to determine MI status and onset date
4. **Pipeline**: Orchestrates the entire process and generates reports

## Output Format

Results are provided in both JSON and CSV formats, including:
- MI detection (Y/N)
- Confidence score
- Onset date (if applicable)
- Supporting evidence from lab results and clinical notes

## Customization

The system can be extended to handle additional clinical variables by implementing new evidence collectors and rule sets following the same pattern.

## License

This project is licensed under the MIT License.
