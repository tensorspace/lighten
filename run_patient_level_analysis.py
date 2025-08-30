#!/usr/bin/env python3
"""
Patient-Level MI Analysis Entry Point

This script demonstrates the new patient-level historical analysis capabilities
that process complete patient visit history instead of individual admissions.

Key Features:
- Groups discharge notes by subject_id (patient)
- Analyzes complete visit chronology for each patient
- Aggregates evidence across all patient visits
- Enhanced MI onset date determination using historical context
- Comprehensive Criteria B evaluation across patient timeline
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lighten_ml.pipeline.patient_level_pipeline import PatientLevelClinicalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[
        logging.FileHandler("patient_level_analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for patient-level MI analysis."""
    logger.info("=== PATIENT-LEVEL MI ANALYSIS STARTING ===")

    # Configuration
    config = {
        "data_paths": {
            "lab_events": "data/labevents.csv",
            "lab_items": "data/d_labitems.csv",
            "discharge_notes": "data/discharge_notes_demo.csv",
        },
        "output": {"directory": "output", "filename": "patient_level_mi_analysis.json"},
        "analysis": {
            "max_patients": None,  # Process all patients, or set to limit for testing
            "include_metadata": True,  # Include detailed analysis metadata
            "historical_analysis": True,  # Enable full historical analysis
        },
    }

    # Validate data files exist
    missing_files = []
    for file_type, file_path in config["data_paths"].items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {file_path}")

    if missing_files:
        logger.error("Missing required data files:")
        for missing_file in missing_files:
            logger.error(f"  - {missing_file}")
        logger.error(
            "Please ensure all data files are available before running analysis."
        )
        return 1

    try:
        # Initialize patient-level pipeline
        logger.info("Initializing Patient-Level Clinical Pipeline...")
        pipeline = PatientLevelClinicalPipeline(
            lab_events_path=config["data_paths"]["lab_events"],
            lab_items_path=config["data_paths"]["lab_items"],
            discharge_notes_path=config["data_paths"]["discharge_notes"],
            output_dir=config["output"]["directory"],
            config=config,
        )

        # Get patient overview
        patient_ids = pipeline.patient_history_loader.get_all_patient_ids()
        logger.info(f"Found {len(patient_ids)} patients for analysis")

        # Process patients (limit for testing if specified)
        patients_to_process = patient_ids
        if config["analysis"]["max_patients"]:
            patients_to_process = patient_ids[: config["analysis"]["max_patients"]]
            logger.info(
                f"Limiting analysis to first {len(patients_to_process)} patients for testing"
            )

        # Run patient-level analysis
        logger.info("Starting comprehensive patient-level MI analysis...")
        start_time = datetime.now()

        results = pipeline.process_all_patients()

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Generate analysis summary
        summary = generate_analysis_summary(results, processing_time)
        logger.info("Analysis Summary:")
        for line in summary.split("\n"):
            if line.strip():
                logger.info(f"  {line}")

        # Save results
        output_path = os.path.join(
            config["output"]["directory"], config["output"]["filename"]
        )

        # Ensure output directory exists
        os.makedirs(config["output"]["directory"], exist_ok=True)

        # Prepare final output
        final_output = {
            "analysis_metadata": {
                "analysis_type": "patient_level_historical",
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "total_patients_processed": len(results),
                "configuration": config,
            },
            "summary": summary,
            "results": results,
        }

        # Save to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        logger.info(f"Patient-level analysis results saved to: {output_path}")
        logger.info("=== PATIENT-LEVEL MI ANALYSIS COMPLETED SUCCESSFULLY ===")

        return 0

    except Exception as e:
        logger.error(f"Patient-level analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def generate_analysis_summary(results: dict, processing_time: float) -> str:
    """Generate a comprehensive analysis summary."""
    total_patients = len(results)
    positive_mi_count = 0
    patients_with_onset_dates = 0
    total_visits = 0
    patients_with_multiple_visits = 0

    for patient_id, patient_data in results.items():
        # Count MI positive cases
        mi_data = patient_data.get("Myocardial Infarction", [{}])[0]
        if mi_data.get("value") == "Y":
            positive_mi_count += 1

        # Count patients with onset dates
        onset_data = mi_data.get("Myocardial Infarction Date", [{}])[0]
        if onset_data.get("value"):
            patients_with_onset_dates += 1

        # Count visits and multi-visit patients
        metadata = patient_data.get("_analysis_metadata", {})
        patient_visits = metadata.get("total_visits", 0)
        total_visits += patient_visits
        if patient_visits > 1:
            patients_with_multiple_visits += 1

    # Calculate statistics
    mi_positive_rate = (
        (positive_mi_count / total_patients * 100) if total_patients > 0 else 0
    )
    onset_date_rate = (
        (patients_with_onset_dates / positive_mi_count * 100)
        if positive_mi_count > 0
        else 0
    )
    avg_visits_per_patient = total_visits / total_patients if total_patients > 0 else 0
    multi_visit_rate = (
        (patients_with_multiple_visits / total_patients * 100)
        if total_patients > 0
        else 0
    )

    summary = f"""
PATIENT-LEVEL MI ANALYSIS SUMMARY
=====================================
Processing Time: {processing_time:.2f} seconds
Total Patients Analyzed: {total_patients}
Total Patient Visits: {total_visits}
Average Visits per Patient: {avg_visits_per_patient:.1f}

MI DIAGNOSIS RESULTS:
- MI Positive Cases: {positive_mi_count} ({mi_positive_rate:.1f}%)
- MI Negative Cases: {total_patients - positive_mi_count} ({100 - mi_positive_rate:.1f}%)
- Onset Dates Determined: {patients_with_onset_dates} ({onset_date_rate:.1f}% of positive cases)

HISTORICAL ANALYSIS INSIGHTS:
- Patients with Multiple Visits: {patients_with_multiple_visits} ({multi_visit_rate:.1f}%)
- Single Visit Patients: {total_patients - patients_with_multiple_visits} ({100 - multi_visit_rate:.1f}%)

METHODOLOGY:
- Evidence aggregated across complete patient visit history
- Chronological analysis of symptoms and biomarkers
- Enhanced onset date determination using temporal patterns
- Comprehensive Criteria B evaluation across patient timeline
"""

    return summary.strip()


def demo_single_patient_analysis(patient_id: str = None):
    """Demonstrate detailed analysis for a single patient."""
    logger.info("=== SINGLE PATIENT ANALYSIS DEMO ===")

    # Configuration for demo
    config = {
        "data_paths": {
            "lab_events": "data/labevents.csv",
            "lab_items": "data/d_labitems.csv",
            "discharge_notes": "data/discharge_notes_demo.csv",
        }
    }

    try:
        # Initialize pipeline
        pipeline = PatientLevelClinicalPipeline(
            lab_events_path=config["data_paths"]["lab_events"],
            lab_items_path=config["data_paths"]["lab_items"],
            discharge_notes_path=config["data_paths"]["discharge_notes"],
        )

        # Get a patient ID if not provided
        if not patient_id:
            patient_ids = pipeline.patient_history_loader.get_all_patient_ids()
            if patient_ids:
                patient_id = patient_ids[0]  # Use first patient for demo
                logger.info(f"Using patient {patient_id} for demo analysis")
            else:
                logger.error("No patients found in dataset")
                return

        # Get patient visit summary
        visit_summary = pipeline.patient_history_loader.get_patient_visit_summary(
            patient_id
        )
        logger.info(f"Patient {patient_id} Visit Summary:")
        logger.info(f"  Total Visits: {visit_summary['total_visits']}")
        if visit_summary["date_range"]:
            logger.info(
                f"  Date Range: {visit_summary['date_range']['first_visit']} to {visit_summary['date_range']['last_visit']}"
            )
            logger.info(f"  Span: {visit_summary['date_range']['span_days']} days")

        # Process the patient
        result = pipeline.process_patient(patient_id)

        # Display detailed results
        logger.info(f"=== DETAILED RESULTS FOR PATIENT {patient_id} ===")
        logger.info(json.dumps(result, indent=2, default=str))

    except Exception as e:
        logger.error(f"Single patient demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patient-Level MI Analysis")
    parser.add_argument("--demo", action="store_true", help="Run single patient demo")
    parser.add_argument("--patient-id", type=str, help="Specific patient ID for demo")

    args = parser.parse_args()

    if args.demo:
        demo_single_patient_analysis(args.patient_id)
    else:
        exit_code = main()
        sys.exit(exit_code)
