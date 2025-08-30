"""Patient-level clinical pipeline for historical analysis."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..data_loaders import LabDataLoader, PatientHistoryLoader
from ..evidence_collectors import (
    AngiographyEvidenceExtractor,
    ClinicalEvidenceExtractor,
    ECGEvidenceExtractor,
    ImagingEvidenceExtractor,
    TroponinAnalyzer,
)
from ..llm_client import LightenLLMClient
from ..resolvers.onset_date_resolver import OnsetDateResolver
from ..rule_engines import MIRuleEngine, MIRuleEngineConfig

logger = logging.getLogger(__name__)


class PatientLevelClinicalPipeline:
    """Enhanced pipeline for patient-level historical MI analysis."""

    def __init__(
        self,
        lab_events_path: str,
        lab_items_path: str,
        discharge_notes_path: str,
        output_dir: str = "output",
        config: Optional[Dict[str, Any]] = None,
    ):
        logger.info("Initializing Patient-Level Clinical Pipeline...")

        # Initialize data loaders
        self.lab_data_loader = LabDataLoader(lab_events_path, lab_items_path)
        self.patient_history_loader = PatientHistoryLoader(discharge_notes_path)

        # Initialize evidence collectors
        self.troponin_analyzer = TroponinAnalyzer()
        self.clinical_extractor = ClinicalEvidenceExtractor()
        self.ecg_extractor = ECGEvidenceExtractor()
        self.imaging_extractor = ImagingEvidenceExtractor()
        self.angiography_extractor = AngiographyEvidenceExtractor()

        # Initialize LLM client and resolvers
        self.llm_client = LightenLLMClient()
        self.onset_date_resolver = OnsetDateResolver(self.llm_client)

        # Initialize rule engine
        rule_config = MIRuleEngineConfig()
        if config and "rule_engine" in config:
            rule_config.update(config["rule_engine"])
        self.rule_engine = MIRuleEngine(rule_config)

        self.output_dir = output_dir
        self.config = config or {}

    def process_patient(self, patient_id: str) -> Dict[str, Any]:
        """Process a single patient's complete visit history for MI analysis.

        Args:
            patient_id: Patient subject_id

        Returns:
            Complete MI analysis results for the patient
        """
        logger.info(f"=== PROCESSING PATIENT {patient_id} ===")
        logger.debug(f"[DEBUG] Starting patient-level analysis for {patient_id}")

        # Get patient's complete visit history
        logger.debug(f"[DEBUG] Loading visit history for patient {patient_id}")
        visit_history = self.patient_history_loader.get_patient_visit_history(
            patient_id
        )
        visit_summary = self.patient_history_loader.get_patient_visit_summary(
            patient_id
        )

        if not visit_history:
            logger.warning(f"[WARNING] No visit history found for patient {patient_id}")
            logger.debug(
                f"[DEBUG] Returning empty result for patient {patient_id} - no visits"
            )
            return self._create_empty_result(patient_id, "No visit history available")

        logger.info(
            f"[PATIENT_INFO] Patient {patient_id} has {visit_summary['total_visits']} visits"
        )
        logger.info(
            f"[PATIENT_INFO] Visit date range: {visit_summary['date_range']['first_visit']} to {visit_summary['date_range']['last_visit']}"
        )
        logger.debug(
            f"[DEBUG] Visit span: {visit_summary['date_range']['span_days']} days"
        )
        logger.debug(f"[DEBUG] Admission IDs: {visit_summary['admission_ids']}")

        try:
            # Step 1: Collect evidence across all visits
            logger.info(
                f"[STEP_1] Collecting historical evidence for patient {patient_id}"
            )
            logger.info(
                f"[PIPELINE_FLOW] Starting cross-admission evidence aggregation for {len(visit_history)} visits"
            )
            logger.debug(
                f"[DEBUG] Visit sequence: {[f"{v['hadm_id']}({v['chartdate']})" for v in visit_history]}"
            )

            historical_evidence = self._collect_historical_evidence(
                patient_id, visit_history
            )

            # Log comprehensive evidence summary
            logger.info(
                f"[EVIDENCE_SUMMARY] Patient {patient_id} - Cross-admission evidence collected:"
            )
            logger.info(
                f"[EVIDENCE_SUMMARY]   Troponin tests: {len(historical_evidence.get('troponin', {}).get('troponin_tests', []))}"
            )
            logger.info(
                f"[EVIDENCE_SUMMARY]   Clinical symptoms: {len(historical_evidence.get('clinical', {}).get('symptoms', []))}"
            )
            logger.info(
                f"[EVIDENCE_SUMMARY]   Clinical diagnoses: {len(historical_evidence.get('clinical', {}).get('diagnoses', []))}"
            )
            logger.info(
                f"[EVIDENCE_SUMMARY]   ECG findings: {len(historical_evidence.get('ecg', {}).get('findings', []))}"
            )
            logger.info(
                f"[EVIDENCE_SUMMARY]   Imaging findings: {len(historical_evidence.get('imaging', {}).get('findings', []))}"
            )
            logger.info(
                f"[EVIDENCE_SUMMARY]   Angiography findings: {len(historical_evidence.get('angiography', {}).get('findings', []))}"
            )

            logger.debug(
                f"[DEBUG] Evidence collection completed for patient {patient_id}"
            )

            # Step 2: Evaluate MI criteria using complete patient history
            logger.info(f"[STEP_2] Evaluating MI criteria for patient {patient_id}")
            logger.info(
                f"[PIPELINE_FLOW] Using aggregated cross-admission evidence for MI diagnosis"
            )
            logger.debug(
                f"[DEBUG] Evidence keys available: {list(historical_evidence.keys())}"
            )

            # Log evidence quality before evaluation
            troponin_available = historical_evidence.get("troponin", {}).get(
                "troponin_available", False
            )
            clinical_evidence_count = len(
                historical_evidence.get("clinical", {}).get("symptoms", [])
            ) + len(historical_evidence.get("clinical", {}).get("diagnoses", []))
            logger.info(
                f"[PRE_EVALUATION] Troponin data available: {troponin_available}"
            )
            logger.info(
                f"[PRE_EVALUATION] Clinical evidence items: {clinical_evidence_count}"
            )

            mi_result = self.rule_engine.evaluate(historical_evidence)

            logger.info(
                f"[MI_RESULT] Patient {patient_id} - MI Criteria Result: {'PASSED' if mi_result.passed else 'FAILED'}"
            )
            logger.info(
                f"[MI_RESULT] Criteria A (Troponin): {'MET' if mi_result.details.get('criteria_A', {}).get('met', False) else 'NOT MET'}"
            )
            logger.info(
                f"[MI_RESULT] Criteria B (Clinical): {'MET' if mi_result.details.get('criteria_B', {}).get('met', False) else 'NOT MET'}"
            )
            logger.debug(f"[DEBUG] MI result details: {mi_result.details}")

            # Step 3: Determine MI onset date using complete visit history
            onset_date = None
            if mi_result.passed:
                logger.info(
                    f"[STEP_3] Determining MI onset date for patient {patient_id}"
                )
                logger.info(
                    f"[PIPELINE_FLOW] Using complete patient timeline for onset date resolution"
                )
                logger.info(
                    f"[TIMELINE_ANALYSIS] Analyzing {len(visit_history)} visits spanning {visit_summary['date_range']['span_days']} days"
                )

                onset_date = self._determine_historical_onset_date(
                    patient_id, visit_history, historical_evidence
                )

                if onset_date:
                    logger.info(
                        f"[ONSET_DATE] Patient {patient_id} - MI Onset Date: {onset_date}"
                    )
                    # Calculate onset relative to first visit
                    first_visit_date = (
                        visit_history[0]["chartdate"] if visit_history else None
                    )
                    if first_visit_date:
                        logger.info(
                            f"[ONSET_CONTEXT] Onset date relative to first visit ({first_visit_date})"
                        )
                else:
                    logger.warning(
                        f"[WARNING] Could not determine onset date for patient {patient_id}"
                    )
                    logger.warning(
                        f"[WARNING] Checked {len(visit_history)} visits but no definitive onset found"
                    )
            else:
                logger.info(
                    f"[STEP_3] Skipping onset date determination - MI criteria not met for patient {patient_id}"
                )
                logger.debug(
                    f"[DEBUG] MI diagnosis negative, onset date not applicable"
                )

            # Step 4: Compile comprehensive results
            logger.info(f"[STEP_4] Compiling results for patient {patient_id}")
            result = self._compile_patient_results(
                patient_id, visit_history, mi_result, onset_date, historical_evidence
            )

            logger.info(
                f"[FINAL_RESULT] Patient {patient_id} - MI Diagnosis: {'POSITIVE' if mi_result.passed else 'NEGATIVE'}"
            )
            if onset_date:
                logger.info(
                    f"[FINAL_RESULT] Patient {patient_id} - MI Onset Date: {onset_date}"
                )

            logger.debug(
                f"[DEBUG] Successfully completed analysis for patient {patient_id}"
            )
            return result

        except Exception as e:
            logger.error(f"[ERROR] Error processing patient {patient_id}: {e}")
            logger.debug(
                f"[DEBUG] Exception details for patient {patient_id}", exc_info=True
            )
            return self._create_empty_result(patient_id, f"Processing error: {str(e)}")

    def _collect_historical_evidence(
        self, patient_id: str, visit_history: List[Dict]
    ) -> Dict[str, Any]:
        """Collect and aggregate evidence across all patient visits.

        Args:
            patient_id: Patient subject_id
            visit_history: Chronologically ordered visit history

        Returns:
            Aggregated evidence dictionary
        """
        logger.info(f"[{patient_id}] === HISTORICAL EVIDENCE COLLECTION ===")
        logger.debug(
            f"[DEBUG] {patient_id} - Starting evidence collection across {len(visit_history)} visits"
        )

        # Initialize aggregated evidence structure
        historical_evidence = {
            "troponin": {"troponin_available": False, "troponin_tests": []},
            "clinical": {"symptoms": [], "diagnoses": []},
            "ecg": {"findings": []},
            "imaging": {"findings": []},
            "angiography": {"findings": []},
            "visit_metadata": {
                "total_visits": len(visit_history),
                "visit_dates": [visit["chartdate"] for visit in visit_history],
                "admission_ids": [visit["hadm_id"] for visit in visit_history],
            },
        }

        logger.debug(
            f"[DEBUG] {patient_id} - Initialized evidence structure with {len(historical_evidence)} categories"
        )

        # Process each visit chronologically
        for visit_idx, visit in enumerate(visit_history, 1):
            hadm_id = visit["hadm_id"]
            visit_date = visit["chartdate"]
            text_length = len(visit.get("text", ""))

            logger.info(
                f"[{patient_id}] === PROCESSING VISIT {visit_idx}/{len(visit_history)} ==="
            )
            logger.info(
                f"[{patient_id}] Visit Details: {hadm_id} on {visit_date} ({text_length:,} chars)"
            )
            logger.debug(
                f"[DEBUG] {patient_id} - Visit {visit_idx} chronological position in patient timeline"
            )

            # Collect troponin data for this visit
            logger.debug(
                f"[DEBUG] {patient_id} - Collecting troponin evidence for visit {hadm_id}"
            )
            visit_troponin = self._collect_visit_troponin(patient_id, hadm_id)

            if visit_troponin["troponin_available"]:
                troponin_count_before = len(
                    historical_evidence["troponin"]["troponin_tests"]
                )
                historical_evidence["troponin"]["troponin_available"] = True
                historical_evidence["troponin"]["troponin_tests"].extend(
                    visit_troponin["troponin_tests"]
                )
                troponin_count_after = len(
                    historical_evidence["troponin"]["troponin_tests"]
                )
                new_tests = troponin_count_after - troponin_count_before

                logger.info(
                    f"[{patient_id}] Visit {hadm_id}: Added {new_tests} troponin tests (total: {troponin_count_after})"
                )

                # Log troponin values for this visit
                for test in visit_troponin["troponin_tests"]:
                    logger.debug(
                        f"[DEBUG] {patient_id} - Troponin: {test.get('value', 'N/A')} {test.get('unit', '')} at {test.get('charttime', 'N/A')} (threshold: {'PASS' if test.get('above_threshold') else 'FAIL'})"
                    )
            else:
                logger.info(f"[{patient_id}] Visit {hadm_id}: No troponin tests found")

            # Collect clinical evidence for this visit
            logger.debug(
                f"[DEBUG] {patient_id} - Collecting clinical evidence for visit {hadm_id}"
            )
            visit_clinical = self._collect_visit_clinical_evidence(
                patient_id, hadm_id, visit["text"]
            )

            if visit_clinical["symptoms"]:
                symptoms_count_before = len(historical_evidence["clinical"]["symptoms"])
                historical_evidence["clinical"]["symptoms"].extend(
                    visit_clinical["symptoms"]
                )
                symptoms_count_after = len(historical_evidence["clinical"]["symptoms"])
                new_symptoms = symptoms_count_after - symptoms_count_before

                logger.info(
                    f"[{patient_id}] Visit {hadm_id}: Added {new_symptoms} symptoms (total: {symptoms_count_after})"
                )

                # Log specific symptoms found
                for symptom in visit_clinical["symptoms"]:
                    logger.debug(
                        f"[DEBUG] {patient_id} - Symptom: {symptom.get('symptom', 'N/A')} (confidence: {symptom.get('confidence', 'N/A')})"
                    )
            else:
                logger.info(f"[{patient_id}] Visit {hadm_id}: No symptoms found")

            if visit_clinical["diagnoses"]:
                diagnoses_count_before = len(
                    historical_evidence["clinical"]["diagnoses"]
                )
                historical_evidence["clinical"]["diagnoses"].extend(
                    visit_clinical["diagnoses"]
                )
                diagnoses_count_after = len(
                    historical_evidence["clinical"]["diagnoses"]
                )
                new_diagnoses = diagnoses_count_after - diagnoses_count_before

                logger.info(
                    f"[{patient_id}] Visit {hadm_id}: Added {new_diagnoses} diagnoses (total: {diagnoses_count_after})"
                )

                # Log specific diagnoses found
                for diagnosis in visit_clinical["diagnoses"]:
                    logger.debug(
                        f"[DEBUG] {patient_id} - Diagnosis: {diagnosis.get('diagnosis', 'N/A')} (confidence: {diagnosis.get('confidence', 'N/A')})"
                    )
            else:
                logger.info(f"[{patient_id}] Visit {hadm_id}: No diagnoses found")

            # Log visit processing completion
            logger.info(f"[{patient_id}] Visit {visit_idx} processing completed")

            # Collect other evidence types (ECG, imaging, angiography)
            # Note: These would be expanded based on available data sources
            logger.debug(
                f"[DEBUG] {patient_id} - Additional evidence collection (ECG, imaging, angiography) would be implemented here"
            )

            # Log visit completion with running totals
            running_totals = {
                "troponin_tests": len(
                    historical_evidence["troponin"]["troponin_tests"]
                ),
                "symptoms": len(historical_evidence["clinical"]["symptoms"]),
                "diagnoses": len(historical_evidence["clinical"]["diagnoses"]),
            }
            logger.debug(
                f"[DEBUG] {patient_id} - Running totals after visit {visit_idx}: T={running_totals['troponin_tests']}, S={running_totals['symptoms']}, D={running_totals['diagnoses']}"
            )

        # Aggregate and deduplicate evidence
        logger.info(f"[{patient_id}] === EVIDENCE AGGREGATION & DEDUPLICATION ===")
        evidence_before_aggregation = {
            "troponin_tests": len(historical_evidence["troponin"]["troponin_tests"]),
            "symptoms": len(historical_evidence["clinical"]["symptoms"]),
            "diagnoses": len(historical_evidence["clinical"]["diagnoses"]),
        }

        historical_evidence = self._aggregate_evidence(historical_evidence)

        evidence_after_aggregation = {
            "troponin_tests": len(historical_evidence["troponin"]["troponin_tests"]),
            "symptoms": len(historical_evidence["clinical"]["symptoms"]),
            "diagnoses": len(historical_evidence["clinical"]["diagnoses"]),
        }

        logger.info(f"[{patient_id}] CROSS-ADMISSION EVIDENCE AGGREGATION COMPLETE:")
        logger.info(
            f"[{patient_id}]   Troponin tests: {evidence_before_aggregation['troponin_tests']} → {evidence_after_aggregation['troponin_tests']} (deduplicated: {evidence_before_aggregation['troponin_tests'] - evidence_after_aggregation['troponin_tests']})"
        )
        logger.info(
            f"[{patient_id}]   Clinical symptoms: {evidence_before_aggregation['symptoms']} → {evidence_after_aggregation['symptoms']} (deduplicated: {evidence_before_aggregation['symptoms'] - evidence_after_aggregation['symptoms']})"
        )
        logger.info(
            f"[{patient_id}]   Clinical diagnoses: {evidence_before_aggregation['diagnoses']} → {evidence_after_aggregation['diagnoses']} (deduplicated: {evidence_before_aggregation['diagnoses'] - evidence_after_aggregation['diagnoses']})"
        )

        # Log timeline span
        visit_dates = historical_evidence.get("visit_metadata", {}).get(
            "visit_dates", []
        )
        if len(visit_dates) > 1:
            timeline_span = (
                (visit_dates[-1] - visit_dates[0]).days
                if hasattr(visit_dates[0], "days")
                else "N/A"
            )
            logger.info(
                f"[{patient_id}]   Timeline span: {timeline_span} days across {len(visit_dates)} visits"
            )

        logger.info(f"[{patient_id}] READY FOR MI CRITERIA EVALUATION")

        return historical_evidence

    def _collect_visit_troponin(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect troponin evidence for a specific visit."""
        logger.debug(
            f"[DEBUG] {patient_id} - Collecting troponin evidence for visit {hadm_id}"
        )

        try:
            # Get troponin tests for this specific admission
            logger.debug(
                f"[DEBUG] {patient_id} - Querying troponin tests for admission {hadm_id}"
            )
            troponin_tests = self.lab_data_loader.get_troponin_tests(
                patient_id, hadm_id
            )

            if not troponin_tests.empty:
                logger.debug(
                    f"[DEBUG] {patient_id} - Found {len(troponin_tests)} troponin tests for visit {hadm_id}"
                )

                # Process troponin data using existing analyzer
                logger.debug(
                    f"[DEBUG] {patient_id} - Analyzing troponin data for visit {hadm_id}"
                )
                troponin_evidence = self.troponin_analyzer.analyze_troponin_data(
                    patient_id, hadm_id, troponin_tests
                )

                logger.debug(
                    f"[DEBUG] {patient_id} - Troponin analysis completed for visit {hadm_id}"
                )
                return troponin_evidence
            else:
                logger.debug(
                    f"[DEBUG] {patient_id} - No troponin tests found for visit {hadm_id}"
                )
                return {"troponin_available": False, "troponin_tests": []}

        except Exception as e:
            logger.error(
                f"[ERROR] {patient_id} - Error collecting troponin for visit {hadm_id}: {e}"
            )
            logger.debug(f"[DEBUG] {patient_id} - Exception details", exc_info=True)
            return {"troponin_available": False, "troponin_tests": []}

    def _collect_visit_clinical_evidence(
        self, patient_id: str, hadm_id: str, clinical_text: str
    ) -> Dict[str, Any]:
        """Collect clinical evidence for a specific visit."""
        logger.debug(
            f"[DEBUG] {patient_id} - Collecting clinical evidence for visit {hadm_id}"
        )

        try:
            # Log clinical text statistics
            text_length = len(clinical_text) if clinical_text else 0
            logger.debug(
                f"[DEBUG] {patient_id} - Clinical text length for visit {hadm_id}: {text_length} characters"
            )

            if text_length == 0:
                logger.warning(
                    f"[WARNING] {patient_id} - Empty clinical text for visit {hadm_id}"
                )
                return {"symptoms": [], "diagnoses": []}

            # Extract clinical evidence from this visit's notes
            logger.debug(
                f"[DEBUG] {patient_id} - Extracting clinical evidence from visit {hadm_id}"
            )
            clinical_evidence = self.clinical_extractor.extract_evidence(
                patient_id, hadm_id, clinical_text
            )

            # Log extraction results
            symptoms_count = len(clinical_evidence.get("symptoms", []))
            diagnoses_count = len(clinical_evidence.get("diagnoses", []))
            logger.debug(
                f"[DEBUG] {patient_id} - Visit {hadm_id} extraction: {symptoms_count} symptoms, {diagnoses_count} diagnoses"
            )

            return clinical_evidence

        except Exception as e:
            logger.error(
                f"[ERROR] {patient_id} - Error collecting clinical evidence for visit {hadm_id}: {e}"
            )
            logger.debug(f"[DEBUG] {patient_id} - Exception details", exc_info=True)
            return {"symptoms": [], "diagnoses": []}

    def _aggregate_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate and deduplicate evidence across visits."""
        logger.debug(f"[DEBUG] Starting evidence aggregation and deduplication")

        # Sort troponin tests chronologically
        troponin_count = len(evidence["troponin"]["troponin_tests"])
        if evidence["troponin"]["troponin_tests"]:
            logger.debug(
                f"[DEBUG] Sorting {troponin_count} troponin tests chronologically"
            )
            evidence["troponin"]["troponin_tests"].sort(
                key=lambda x: x.get("timestamp", ""), reverse=False
            )
            logger.debug(f"[DEBUG] Troponin tests sorted successfully")

        # Deduplicate symptoms (keep unique symptom-context combinations)
        original_symptoms_count = len(evidence["clinical"]["symptoms"])
        logger.debug(f"[DEBUG] Deduplicating {original_symptoms_count} symptoms")

        unique_symptoms = []
        seen_symptoms = set()
        for symptom in evidence["clinical"]["symptoms"]:
            symptom_key = f"{symptom.get('symptom', '')}-{symptom.get('context', '')}"
            if symptom_key not in seen_symptoms:
                unique_symptoms.append(symptom)
                seen_symptoms.add(symptom_key)
        evidence["clinical"]["symptoms"] = unique_symptoms

        logger.debug(
            f"[DEBUG] Symptoms after deduplication: {len(unique_symptoms)} (removed {original_symptoms_count - len(unique_symptoms)} duplicates)"
        )

        # Deduplicate diagnoses
        original_diagnoses_count = len(evidence["clinical"]["diagnoses"])
        logger.debug(f"[DEBUG] Deduplicating {original_diagnoses_count} diagnoses")

        unique_diagnoses = []
        seen_diagnoses = set()
        for diagnosis in evidence["clinical"]["diagnoses"]:
            diag_key = diagnosis.get("diagnosis", "")
            if diag_key not in seen_diagnoses:
                unique_diagnoses.append(diagnosis)
                seen_diagnoses.add(diag_key)
        evidence["clinical"]["diagnoses"] = unique_diagnoses

        logger.debug(
            f"[DEBUG] Diagnoses after deduplication: {len(unique_diagnoses)} (removed {original_diagnoses_count - len(unique_diagnoses)} duplicates)"
        )
        logger.debug(f"[DEBUG] Evidence aggregation completed successfully")

        return evidence

    def _determine_historical_onset_date(
        self, patient_id: str, visit_history: List[Dict], evidence: Dict[str, Any]
    ) -> Optional[str]:
        """Determine MI onset date using complete patient history."""
        try:
            # Compile all clinical texts for comprehensive analysis
            all_clinical_texts = []
            for visit in visit_history:
                all_clinical_texts.append(
                    {
                        "hadm_id": visit["hadm_id"],
                        "chartdate": visit["chartdate"],
                        "text": visit["text"],
                    }
                )

            # Use enhanced onset date resolver with historical context
            onset_date = self.onset_date_resolver.resolve_onset_date_with_history(
                patient_id, all_clinical_texts, evidence
            )

            return onset_date

        except Exception as e:
            logger.error(f"Error determining onset date for patient {patient_id}: {e}")
            return None

    def _compile_patient_results(
        self,
        patient_id: str,
        visit_history: List[Dict],
        mi_result: Any,
        onset_date: Optional[str],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile comprehensive patient results."""
        # Extract symptoms for output format
        symptoms = []
        if evidence.get("clinical", {}).get("symptoms"):
            symptoms = [
                symptom.get("symptom", "")
                for symptom in evidence["clinical"]["symptoms"]
            ]

        # Create result in required format
        result = {
            patient_id: {
                "Myocardial Infarction": [
                    {
                        "value": "Y" if mi_result.passed else "N",
                        "Myocardial Infarction Date": [
                            {"value": onset_date, "Symptoms": symptoms}
                        ],
                    }
                ]
            }
        }

        # Add detailed analysis metadata
        result[patient_id]["_analysis_metadata"] = {
            "total_visits": len(visit_history),
            "visit_date_range": {
                "first": str(visit_history[0]["chartdate"]) if visit_history else None,
                "last": str(visit_history[-1]["chartdate"]) if visit_history else None,
            },
            "evidence_summary": {
                "troponin_tests": len(
                    evidence.get("troponin", {}).get("troponin_tests", [])
                ),
                "symptoms_found": len(evidence.get("clinical", {}).get("symptoms", [])),
                "diagnoses_found": len(
                    evidence.get("clinical", {}).get("diagnoses", [])
                ),
            },
            "mi_criteria_details": {
                "criteria_a_met": mi_result.details.get("criteria_A", {}).get(
                    "met", False
                ),
                "criteria_b_met": mi_result.details.get("criteria_B", {}).get(
                    "met", False
                ),
                "overall_passed": mi_result.passed,
            },
        }

        return result

    def _create_empty_result(self, patient_id: str, reason: str) -> Dict[str, Any]:
        """Create empty result for patients with no data or errors."""
        return {
            patient_id: {
                "Myocardial Infarction": [
                    {
                        "value": "N",
                        "Myocardial Infarction Date": [{"value": None, "Symptoms": []}],
                    }
                ],
                "_analysis_metadata": {"error": reason, "total_visits": 0},
            }
        }

    def process_all_patients(self) -> Dict[str, Any]:
        """Process all patients in the dataset."""
        logger.info("=== PROCESSING ALL PATIENTS ===")

        # Get all patient IDs
        patient_ids = self.patient_history_loader.get_all_patient_ids()
        logger.info(f"Found {len(patient_ids)} patients to process")

        all_results = {}

        for idx, patient_id in enumerate(patient_ids, 1):
            logger.info(f"Processing patient {idx}/{len(patient_ids)}: {patient_id}")

            try:
                patient_result = self.process_patient(patient_id)
                all_results.update(patient_result)

            except Exception as e:
                logger.error(f"Failed to process patient {patient_id}: {e}")
                all_results.update(
                    self._create_empty_result(
                        patient_id, f"Processing failed: {str(e)}"
                    )
                )

        logger.info(f"Completed processing {len(all_results)} patients")
        return all_results
