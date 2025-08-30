# Patient-Level MI Diagnosis Analysis

## Overview

This document describes the patient-level myocardial infarction (MI) diagnosis analysis system, which extends the original admission-level analysis to provide comprehensive historical analysis across all patient visits.

## Architecture

### Core Components

#### 1. PatientHistoryLoader
- **Purpose**: Load and manage patient visit history from discharge notes
- **Key Features**:
  - Groups discharge notes by `subject_id` (patient ID)
  - Sorts visits chronologically by `chartdate`
  - Provides caching for efficient repeated access
  - Comprehensive debug logging for data loading visibility

#### 2. PatientLevelClinicalPipeline
- **Purpose**: Process complete patient visit history for MI analysis
- **Key Features**:
  - Aggregates evidence across all patient visits
  - Deduplicates symptoms and diagnoses
  - Compiles comprehensive MI diagnosis results
  - Enhanced debug logging for step-by-step tracking

#### 3. Enhanced OnsetDateResolver
- **Purpose**: Determine MI onset date using complete patient timeline
- **Key Features**:
  - Hierarchical decision logic across all visits
  - LLM-based temporal pattern analysis
  - Historical context consideration
  - Optimized logging for clarity

#### 4. Extended LabDataLoader
- **Purpose**: Support patient-level lab data aggregation
- **Key Features**:
  - Patient-level troponin history retrieval
  - Complete lab test history across admissions
  - Patient admission summaries
  - Comprehensive debug logging

## Usage

### Basic Patient-Level Analysis

```python
from lighten_ml.pipeline import PatientLevelClinicalPipeline
from lighten_ml.data_loaders import PatientHistoryLoader, LabDataLoader

# Initialize components
patient_loader = PatientHistoryLoader("discharge_notes.csv")
lab_loader = LabDataLoader("labevents.csv", "d_labitems.csv")

# Create patient-level pipeline
pipeline = PatientLevelClinicalPipeline(
    patient_history_loader=patient_loader,
    lab_data_loader=lab_loader
)

# Process a patient
patient_id = "12345"
result = pipeline.process_patient(patient_id)

print(f"MI Diagnosis: {result['mi_diagnosis']}")
print(f"Onset Date: {result['onset_date']}")
print(f"Evidence Summary: {result['evidence_summary']}")
```

### Command-Line Interface

```bash
# Run demo analysis for a single patient
python run_patient_level_analysis.py --demo --patient-id 12345

# Process full dataset
python run_patient_level_analysis.py --full-run

# Enable debug logging
python run_patient_level_analysis.py --demo --patient-id 12345 --log-level DEBUG
```

## Data Requirements

### Input Files

1. **Discharge Notes CSV**
   - Required columns: `subject_id`, `hadm_id`, `chartdate`, `text`
   - Format: Patient visits grouped by subject_id, sorted chronologically

2. **Lab Events CSV**
   - Standard MIMIC lab events format
   - Required for troponin and other lab test analysis

3. **Lab Items CSV**
   - Standard MIMIC lab items format
   - Maps itemids to lab test names

### Expected Data Structure

```
Patient Visit History:
├── subject_id: "12345"
├── visits:
│   ├── Visit 1: hadm_id="100001", chartdate="2020-01-15", text="..."
│   ├── Visit 2: hadm_id="100002", chartdate="2020-03-20", text="..."
│   └── Visit 3: hadm_id="100003", chartdate="2020-06-10", text="..."
```

## Analysis Process

### 1. Patient History Collection
- Load all visits for the patient
- Sort chronologically by chart date
- Cache for efficient repeated access

### 2. Evidence Aggregation
- Collect troponin data across all visits
- Extract clinical evidence from all discharge notes
- Aggregate and deduplicate symptoms/diagnoses
- Maintain chronological ordering

### 3. MI Criteria Evaluation
- Apply MI rule engine to aggregated evidence
- Consider historical context and patterns
- Generate comprehensive diagnosis result

### 4. Onset Date Determination
- Analyze complete patient timeline
- Use hierarchical decision logic:
  1. LLM-based temporal analysis (if available)
  2. Earliest symptom onset
  3. Earliest troponin elevation
  4. First visit date (fallback)

## Logging and Debugging

### Log Levels
- **INFO**: High-level process steps and results
- **DEBUG**: Detailed step-by-step analysis
- **WARNING**: Data quality issues or fallbacks
- **ERROR**: Processing failures with context

### Log Prefixes
- `[PATIENT_ANALYSIS]`: Patient-level processing steps
- `[DATA_LOADING]`: Data loading operations
- `[TROPONIN_HISTORY]`: Troponin data collection
- `[LAB_HISTORY]`: Lab data operations
- `[ONSET_RESOLUTION]`: Onset date determination
- `[DEBUG]`: Detailed debugging information

### Example Debug Output
```
[PATIENT_ANALYSIS] 12345 - Starting patient-level MI analysis
[DATA_LOADING] Loading patient history from discharge_notes.csv
[VISIT_HISTORY] Patient 12345: 3 visits from 2020-01-15 to 2020-06-10
[TROPONIN_HISTORY] 12345 - Found 5 troponin tests across 2 admissions
[ONSET_RESOLUTION] 12345 - Resolving MI onset date using complete patient history
[PATIENT_ANALYSIS] 12345 - MI diagnosis: POSITIVE, Onset: 2020-03-20
```

## Performance Considerations

### Optimization Features
- **Caching**: Patient visit history cached for repeated access
- **Efficient Filtering**: Optimized database queries for patient-specific data
- **Batch Processing**: Support for processing multiple patients
- **Memory Management**: Efficient data structures for large datasets

### Scalability
- Designed for processing thousands of patients
- Configurable batch sizes for memory management
- Optional parallel processing support
- Progress tracking for long-running analyses

## Comparison: Admission-Level vs Patient-Level

| Aspect | Admission-Level | Patient-Level |
|--------|----------------|---------------|
| **Scope** | Single hospital visit | Complete patient history |
| **Evidence** | One admission's data | Aggregated across all visits |
| **Onset Date** | Visit-specific analysis | Historical timeline analysis |
| **Accuracy** | Limited by single visit | Enhanced by complete context |
| **Use Case** | Real-time diagnosis | Comprehensive retrospective analysis |

## Error Handling

### Common Issues and Solutions

1. **Missing Patient Data**
   - Graceful fallback with informative logging
   - Returns empty results with clear status

2. **Data Quality Issues**
   - Validation with detailed error messages
   - Continues processing with available data

3. **LLM Service Unavailable**
   - Falls back to rule-based onset date resolution
   - Logs fallback reason for transparency

## Future Enhancements

### Planned Features
- **Multi-condition Analysis**: Extend beyond MI to other cardiac conditions
- **Risk Stratification**: Patient risk scoring based on historical patterns
- **Outcome Prediction**: Predict future MI risk using historical data
- **Integration APIs**: RESTful APIs for external system integration

### Research Opportunities
- **Temporal Pattern Mining**: Advanced ML for symptom progression patterns
- **Comparative Effectiveness**: Compare patient-level vs admission-level accuracy
- **Population Health**: Aggregate insights across patient populations

## Support and Troubleshooting

### Debug Mode
Enable comprehensive logging for troubleshooting:
```bash
python run_patient_level_analysis.py --demo --patient-id 12345 --log-level DEBUG
```

### Common Troubleshooting Steps
1. Check data file paths and formats
2. Verify patient ID exists in dataset
3. Review log output for specific error messages
4. Ensure all required dependencies are installed

### Contact Information
For technical support or questions about the patient-level analysis system, please refer to the project documentation or contact the development team.
