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
        self.troponin_analyzer = TroponinAnalyzer(self.lab_data_loader)
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
        """Process a single patient's complete visit history for MI analysis."""
        logger.info(f"[PIPELINE_START] === PROCESSING PATIENT {patient_id} ===")

        # Get patient's complete visit history
        visit_history = self.patient_history_loader.get_patient_visit_history(patient_id)
        if not visit_history:
            logger.warning(f"[{patient_id}] No visit history found. Skipping.")
            return self._create_empty_result(patient_id, "No visit history available")

        visit_summary = self.patient_history_loader.get_patient_visit_summary(patient_id)
        logger.info(f"[{patient_id}] Found {visit_summary['total_visits']} visits from {visit_summary['date_range']['first_visit']} to {visit_summary['date_range']['last_visit']}")

        try:
            # Step 1: Collect and analyze patient's full troponin history for Criteria A
            logger.info(f"[{patient_id}] [STEP 1/3] Collecting full troponin history...")
            troponin_evidence = self._collect_patient_troponin_history(patient_id, visit_history)

            # Step 2: Evaluate Criteria A (Troponin)
            logger.info(f"[{patient_id}] [STEP 2/3] Evaluating Criteria A (Troponin)...")
            criteria_a_result = self.rule_engine.evaluate_criteria_a(troponin_evidence)

            if not criteria_a_result["met"]:
                logger.info(f"[{patient_id}] Criteria A: NOT MET. Skipping further analysis.")
                mi_result = RuleResult(passed=False, details={'criteria_A': criteria_a_result["details"], 'criteria_B': {'met': False, 'reason': 'Criteria A not met'}})
                result = self._compile_patient_results(patient_id, visit_history, mi_result, None, {"troponin": troponin_evidence})
                logger.info(f"[PIPELINE_END] === PATIENT {patient_id}: MI DIAGNOSIS: NEGATIVE (Criteria A not met) ===")
                return result

            logger.info(f"[{patient_id}] Criteria A: MET. Proceeding to collect clinical evidence.")

            # Step 3: Collect clinical evidence, evaluate MI, and determine onset date
            logger.info(f"[{patient_id}] [STEP 3/3] Collecting clinical evidence for Criteria B...")
            clinical_evidence = self._collect_historical_clinical_evidence(patient_id, visit_history)

            historical_evidence = {"troponin": troponin_evidence, **clinical_evidence}

            logger.info(f"[{patient_id}] Running final MI evaluation...")
            mi_result = self.rule_engine.evaluate(historical_evidence)

            onset_date = None
            if mi_result.passed:
                logger.info(f"[{patient_id}] MI diagnosis is POSITIVE. Determining onset date...")
                onset_date = self._determine_historical_onset_date(patient_id, visit_history, historical_evidence)
                logger.info(f"[{patient_id}] Determined MI Onset Date: {onset_date}")
            else:
                logger.info(f"[{patient_id}] MI diagnosis is NEGATIVE (Criteria B not met).")

            # Compile and return final results
            result = self._compile_patient_results(patient_id, visit_history, mi_result, onset_date, historical_evidence)
            logger.info(f"[PIPELINE_END] === PATIENT {patient_id}: MI DIAGNOSIS: {'POSITIVE' if mi_result.passed else 'NEGATIVE'} ===")
            return result

        except Exception as e:
            logger.error(f"[{patient_id}] [ERROR] Pipeline failed for patient {patient_id}: {e}", exc_info=True)
            return self._create_empty_result(patient_id, f"Processing error: {str(e)}")

    def _collect_patient_troponin_history(self, patient_id: str, visit_history: List[Dict]) -> Dict[str, Any]:
        """Collect only the troponin history for a patient across all visits."""
        troponin_tests = []
        troponin_available = False

        for visit in visit_history:
            hadm_id = visit["hadm_id"]
            logger.debug(f"[{patient_id}] Checking for troponins in visit {hadm_id}")
            visit_troponin = self._collect_visit_troponin(patient_id, hadm_id)
            if visit_troponin["troponin_available"]:
                troponin_available = True
                troponin_tests.extend(visit_troponin["troponin_tests"])

        logger.info(f"[{patient_id}] Found {len(troponin_tests)} total troponin tests across {len(visit_history)} visits.")
        return {
            "troponin_available": troponin_available,
            "troponin_tests": troponin_tests
        }

    def _collect_historical_clinical_evidence(self, patient_id: str, visit_history: List[Dict]) -> Dict[str, Any]:
        """Collect non-troponin clinical evidence across all patient visits."""
        clinical_evidence = {
            "clinical": {"symptoms": [], "diagnoses": []},
            "ecg": {"findings": []},
            "imaging": {"findings": []},
            "angiography": {"findings": []},
        }

        for visit in visit_history:
            hadm_id = visit["hadm_id"]
            text = visit["text"]
            logger.debug(f"[{patient_id}] Extracting clinical evidence from visit {hadm_id}")

            # Collect clinical evidence
            visit_clinical = self._collect_visit_clinical_evidence(patient_id, hadm_id, text)
            clinical_evidence["clinical"]["symptoms"].extend(visit_clinical["symptoms"])
            clinical_evidence["clinical"]["diagnoses"].extend(visit_clinical["diagnoses"])

            # Placeholder for other evidence collectors (ECG, imaging, etc.)

        symptom_count = len(clinical_evidence["clinical"]["symptoms"])
        diagnosis_count = len(clinical_evidence["clinical"]["diagnoses"])
        logger.info(f"[{patient_id}] Found {symptom_count} total symptoms and {diagnosis_count} total diagnoses.")
        return clinical_evidence

    def _collect_visit_troponin(
        self, patient_id: str, hadm_id: str
    ) -> Dict[str, Any]:
        """Collect troponin evidence for a specific visit."""
        logger.debug(
            f"[DEBUG] {patient_id} - Collecting troponin evidence for visit {hadm_id}"
        )

        try:
            # Get troponin tests for this specific admission
            logger.debug(
                f"[DEBUG] {patient_id} - Querying troponin tests for admission {hadm_id}"
            )
            troponin_df = self.lab_data_loader.get_troponin_tests(
                patient_id, hadm_id
            )  # Limit to 1000 for performance
        except Exception as e:
            logger.error(
                f"[{patient_id}] Failed to get troponin tests for admission {hadm_id}: {e}"
            )
            return {"troponin_available": False, "troponin_tests": []}

        if not troponin_df.empty:
            logger.debug(
                f"[{patient_id}] Found {len(troponin_df)} troponin results for admission {hadm_id}"
            )
            processed_troponins = self.troponin_analyzer.process_troponin_data(
                troponin_df
            )
            return {
                "troponin_available": True,
                "troponin_tests": processed_troponins,
            }
        else:
            return {"troponin_available": False, "troponin_tests": []}

    def _collect_visit_clinical_evidence(
        self, patient_id: str, hadm_id: str, text: str
    ) -> Dict[str, Any]:
        """Collect clinical evidence from a single discharge note."""
        if not text:
            logger.debug(f"[{patient_id}] No text provided for admission {hadm_id}")
            return {"symptoms": [], "diagnoses": []}

        try:
            logger.debug(
                f"[{patient_id}] Extracting clinical evidence from note for admission {hadm_id} ({len(text)} chars)"
            )
            clinical_evidence = self.clinical_extractor.extract(text, hadm_id)
            return clinical_evidence
        except Exception as e:
            logger.error(
                f"[{patient_id}] Failed to extract clinical evidence for admission {hadm_id}: {e}"
            )
            return {"symptoms": [], "diagnoses": []}

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

    def _create_empty_result(self, patient_id: str, error_message: str) -> Dict[str, Any]:
        """Create a standardized empty result for failed processing."""
        return {
            "patient_id": patient_id,
            "mi_diagnosis": "Error",
            "details": {"error": error_message},
            "mi_onset_date": None,
            "evidence": {},
            "metadata": {"processing_status": "Failed"},
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
