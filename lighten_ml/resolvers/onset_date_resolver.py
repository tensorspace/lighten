"""Resolver for determining the onset date of Myocardial Infarction."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
from dateutil.parser import parse as date_parse

logger = logging.getLogger(__name__)


class OnsetDateResolver:
    """Determines the MI onset date based on a hierarchy of evidence."""

    def __init__(self, llm_client=None):
        """Initialize the resolver with optional LLM client for enhanced analysis."""
        self.llm_client = llm_client

    def resolve(
        self, evidence: Dict[str, Any], admission_time: Optional[pd.Timestamp] = None
    ) -> Dict[str, Optional[str]]:
        """Resolve the MI onset date by checking evidence sources in order of priority.

        The hierarchy is:
        1. Symptom Onset Date
        2. First Abnormal ECG Date
        3. First Elevated Troponin Date
        4. Clinical Recognition/Diagnosis Date
        5. Hospital Presentation/Admission Date

        Args:
            evidence: A dictionary containing all collected evidence for the admission.
            admission_time: The timestamp of the hospital admission.

        Returns:
            A dictionary containing the onset date and the rationale.
        """
        logger.info("[ONSET_RESOLUTION] === STARTING MI ONSET DATE RESOLUTION ===")
        logger.info(
            "[ONSET_RESOLUTION] Using hierarchical approach across patient timeline"
        )
        logger.info(
            "[ONSET_RESOLUTION] Priority: Symptoms -> ECG -> Troponin -> Diagnosis -> Admission"
        )

        # Log evidence inventory
        clinical_data = evidence.get("clinical", {})
        troponin_data = evidence.get("troponin", {})
        ecg_data = evidence.get("ecg", {})
        visit_metadata = evidence.get("visit_metadata", {})

        logger.info("[ONSET_RESOLUTION] Cross-admission evidence inventory:")
        logger.info(
            f"[ONSET_RESOLUTION]   Symptoms: {len(clinical_data.get('symptoms', []))}"
        )
        logger.info(
            f"[ONSET_RESOLUTION]   ECG findings: {len(ecg_data.get('ecg_findings', []))}"
        )
        logger.info(
            f"[ONSET_RESOLUTION]   Troponin tests: {len(troponin_data.get('troponin_tests', []))}"
        )
        logger.info(
            f"[ONSET_RESOLUTION]   Diagnoses: {len(clinical_data.get('diagnoses', []))}"
        )
        logger.info(
            f"[ONSET_RESOLUTION]   Total visits: {visit_metadata.get('total_visits', 0)}"
        )

        if admission_time:
            logger.info(
                f"[ONSET_RESOLUTION]   Admission time available: {admission_time}"
            )

        # 1. Symptom Onset Date (from clinical notes)
        logger.info(
            "[ONSET_RESOLUTION] STEP 1: Searching for earliest symptom onset..."
        )
        symptom_onset = self._find_earliest_symptom_date(
            evidence.get("clinical", {}).get("symptoms", [])
        )
        if symptom_onset:
            logger.info(
                f"[ONSET_RESOLUTION] FOUND: Symptom onset date: {symptom_onset}"
            )
            logger.info(
                f"[ONSET_RESOLUTION] Using highest priority evidence (symptoms)"
            )
            return {
                "onset_date": symptom_onset.isoformat(),
                "rationale": "Symptom onset date (highest priority)",
            }
        else:
            logger.info(f"[ONSET_RESOLUTION] No symptom onset dates found")

        # 2. First Abnormal ECG Date
        logger.info("[ONSET_RESOLUTION] STEP 2: Searching for earliest abnormal ECG...")
        ecg_onset = self._find_earliest_ecg_date(
            evidence.get("ecg", {}).get("ecg_findings", [])
        )
        if ecg_onset:
            logger.info(
                f"[ONSET_RESOLUTION] FOUND: First abnormal ECG date: {ecg_onset}"
            )
            logger.info(
                f"[ONSET_RESOLUTION] Using second priority evidence (ECG changes)"
            )
            return {
                "onset_date": ecg_onset.isoformat(),
                "rationale": "First abnormal ECG date (second priority)",
            }
        else:
            logger.info(f"[ONSET_RESOLUTION] No abnormal ECG dates found")

        # 3. First Elevated Troponin Date
        logger.info(
            "[ONSET_RESOLUTION] STEP 3: Searching for first elevated troponin..."
        )
        troponin_tests = evidence.get("troponin", {}).get("troponin_tests", [])
        logger.debug(
            f"[DEBUG] Analyzing {len(troponin_tests)} troponin tests across patient timeline"
        )
        troponin_onset = self._find_first_elevated_troponin_date(troponin_tests)
        if troponin_onset:
            logger.info(
                f"[ONSET_RESOLUTION] FOUND: First elevated troponin date: {troponin_onset}"
            )
            logger.info(
                f"[ONSET_RESOLUTION] Using third priority evidence (troponin elevation)"
            )
            return {
                "onset_date": troponin_onset.isoformat(),
                "rationale": "First elevated troponin date (third priority)",
            }
        else:
            logger.info(f"[ONSET_RESOLUTION] No elevated troponin dates found")

        # 4. Clinical Recognition/Diagnosis Date
        logger.info(
            "[ONSET_RESOLUTION] STEP 4: Searching for earliest diagnosis date..."
        )
        diagnoses = evidence.get("clinical", {}).get("diagnoses", [])
        logger.debug(
            f"[DEBUG] Analyzing {len(diagnoses)} diagnoses across patient timeline"
        )
        diagnosis_date = self._find_earliest_diagnosis_date(diagnoses)
        if diagnosis_date:
            logger.info(
                f"[ONSET_RESOLUTION] FOUND: Clinical diagnosis date: {diagnosis_date}"
            )
            logger.info(
                f"[ONSET_RESOLUTION] Using fourth priority evidence (clinical diagnosis)"
            )
            return {
                "onset_date": diagnosis_date.isoformat(),
                "rationale": "Clinical diagnosis date (fourth priority)",
            }
        else:
            logger.info(f"[ONSET_RESOLUTION] No diagnosis dates found")

        # 5. Hospital Presentation/Admission Date
        logger.info("[ONSET_RESOLUTION] STEP 5: Checking hospital admission date...")
        if admission_time:
            logger.info(
                f"[ONSET_RESOLUTION] FALLBACK: Using admission date: {admission_time}"
            )
            logger.info(
                f"[ONSET_RESOLUTION] Using lowest priority evidence (admission date)"
            )
            return {
                "onset_date": admission_time.isoformat(),
                "rationale": "Hospital admission date (fallback)",
            }
        else:
            logger.info(f"[ONSET_RESOLUTION] No admission time available")

        logger.warning(
            "[ONSET_RESOLUTION] RESOLUTION FAILED: No onset date could be determined"
        )
        logger.warning(
            "[ONSET_RESOLUTION] WARNING: Exhausted all evidence sources across patient timeline"
        )
        return {
            "onset_date": None,
            "rationale": "No definitive onset date could be determined from cross-admission evidence",
        }

    def resolve_onset_date_with_history(
        self, patient_id: str, visit_history: List[Dict], evidence: Dict[str, Any]
    ) -> Optional[str]:
        """Enhanced onset date resolution using complete patient visit history.

        This method analyzes the patient's complete visit history to determine
        the most accurate MI onset date by considering:
        1. Chronological symptom progression across visits
        2. First appearance of cardiac biomarkers
        3. Historical clinical context and prior MI episodes
        4. LLM-based analysis of temporal patterns in clinical notes

        Args:
            patient_id: Patient subject_id
            visit_history: List of all patient visits with clinical texts
            evidence: Aggregated evidence across all visits

        Returns:
            Best estimate of MI onset date or None if undetermined
        """
        logger.info(
            f"[{patient_id}] [ONSET_RESOLUTION] Starting onset date resolution using complete patient history"
        )
        logger.info(
            f"[{patient_id}] [ONSET_RESOLUTION] Analyzing {len(visit_history)} visits for temporal patterns"
        )
        logger.debug(
            f"[{patient_id}] [DEBUG] Starting historical onset date resolution process"
        )

        # Strategy 1: LLM-based analysis of the full timeline
        logger.info(
            f"[{patient_id}] [ONSET_RESOLUTION] Attempting Strategy 1: LLM Analysis"
        )
        try:
            llm_onset_date = self._llm_resolve_onset_date(
                patient_id, visit_history, evidence
            )
            if llm_onset_date:
                logger.info(
                    f"[{patient_id}] [ONSET_RESOLUTION] Strategy 1 SUCCESS. Onset Date: {llm_onset_date}"
                )
                return llm_onset_date
        except Exception as e:
            logger.warning(
                f"[{patient_id}] [ONSET_RESOLUTION] LLM analysis failed: {e}"
            )

        logger.info(
            f"[{patient_id}] [ONSET_RESOLUTION] LLM analysis did not yield a date. Trying fallback strategies."
        )

        # Fallback Strategy 2: Earliest symptom date
        logger.info(
            f"[{patient_id}] [ONSET_RESOLUTION] Attempting Strategy 2: Earliest Symptom Date"
        )
        symptom_date = self._get_earliest_symptom_date(evidence)
        if symptom_date:
            logger.info(
                f"[{patient_id}] [ONSET_RESOLUTION] Strategy 2 SUCCESS. Onset Date: {symptom_date}"
            )
            return symptom_date

        # Fallback Strategy 3: Earliest high troponin date
        logger.info(
            f"[{patient_id}] [ONSET_RESOLUTION] Attempting Strategy 3: Earliest High Troponin Date"
        )
        troponin_date = self._get_earliest_high_troponin_date(evidence)
        if troponin_date:
            logger.info(
                f"[{patient_id}] [ONSET_RESOLUTION] Strategy 3 SUCCESS. Onset Date: {troponin_date}"
            )
            return troponin_date

        logger.warning(
            f"[{patient_id}] [ONSET_RESOLUTION] All strategies failed. Could not determine onset date."
        )
        return None

    def _parse_date(self, date_str: str) -> Optional[Any]:
        try:
            return date_parse(date_str)
        except (ValueError, TypeError):
            return None

    def _find_earliest_symptom_date(
        self, symptoms: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """Find the earliest documented symptom onset date."""
        earliest_date = None
        for symptom in symptoms:
            # Assumes LLM/regex extractor provides an 'onset_time' field
            onset_time_str = symptom.get("onset_time") or symptom.get("charttime")
            if onset_time_str:
                current_date = self._parse_date(onset_time_str)
                if current_date and (
                    earliest_date is None or current_date < earliest_date
                ):
                    earliest_date = current_date
        return earliest_date

    def _find_earliest_ecg_date(
        self, ecg_findings: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """Find the earliest MI-related ECG finding date."""
        earliest_date = None
        mi_related_findings = [f for f in ecg_findings if f.get("mi_related")]
        for finding in mi_related_findings:
            timestamp_str = finding.get("charttime")
            if timestamp_str:
                current_date = self._parse_date(timestamp_str)
                if current_date and (
                    earliest_date is None or current_date < earliest_date
                ):
                    earliest_date = current_date
        return earliest_date

    def _find_earliest_diagnosis_date(
        self, diagnoses: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """Find the earliest MI diagnosis date."""
        earliest_date = None
        for diagnosis in diagnoses:
            date_str = diagnosis.get("diagnosis_date") or diagnosis.get("charttime")
            if date_str:
                current_date = self._parse_date(date_str)
                if current_date and (
                    earliest_date is None or current_date < earliest_date
                ):
                    earliest_date = current_date
        return earliest_date

    def _find_first_elevated_troponin_date(
        self, troponin_tests: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """Find the date of the first troponin above the diagnostic threshold."""
        # Assumes troponin_tests are sorted by time
        for test in troponin_tests:
            if test.get("above_threshold"):
                timestamp_str = test.get("timestamp")
                if timestamp_str:
                    return self._parse_date(timestamp_str)
        return None

    def _llm_resolve_onset_date(
        self, patient_id: str, visit_history: List[Dict], evidence: Dict[str, Any]
    ) -> Optional[str]:
        """Use LLM to analyze the timeline and determine onset date."""
        # This is a placeholder for a complex LLM call
        # In a real implementation, this would format the history and evidence
        # and query the LLM with a sophisticated prompt.
        logger.debug(f"[{patient_id}] Preparing data for LLM onset date analysis.")
        # For demonstration, we'll assume the LLM doesn't find a date
        return None

    def _get_earliest_symptom_date(
        self, evidence: Dict[str, Any]
    ) -> Optional[datetime]:
        """Get the earliest recorded symptom date."""
        symptoms = evidence.get("clinical", {}).get("symptoms", [])
        if not symptoms:
            logger.debug("[ONSET_RESOLVER] No symptoms available for date resolution.")
            return None

        # Assuming symptoms have a 'timestamp' field
        symptom_dates = [s["timestamp"] for s in symptoms if "timestamp" in s]
        if not symptom_dates:
            logger.debug("[ONSET_RESOLVER] Symptoms found, but they lack timestamps.")
            return None

        earliest_date = min(symptom_dates)
        logger.debug(f"[ONSET_RESOLVER] Earliest symptom date found: {earliest_date}")
        return earliest_date

    def _get_earliest_high_troponin_date(
        self, evidence: Dict[str, Any]
    ) -> Optional[datetime]:
        """Get the earliest recorded troponin test above the threshold."""
        troponin_tests = evidence.get("troponin", {}).get("troponin_tests", [])
        high_troponins = [t for t in troponin_tests if t.get("above_threshold")]

        if not high_troponins:
            logger.debug(
                "[ONSET_RESOLVER] No high troponin tests available for date resolution."
            )
            return None

        # Timestamps can be in 'charttime' or 'storetime'
        troponin_dates = []
        for t in high_troponins:
            if "charttime" in t and isinstance(t["charttime"], datetime):
                troponin_dates.append(t["charttime"])

        if not troponin_dates:
            logger.debug(
                "[ONSET_RESOLVER] High troponins found, but they lack charttime timestamps."
            )
            return None

        earliest_date = min(troponin_dates)
        logger.debug(
            f"[ONSET_RESOLVER] Earliest high troponin date found: {earliest_date}"
        )
        return earliest_date

    def _find_earliest_historical_troponin(
        self, evidence: Dict[str, Any]
    ) -> Optional[str]:
        """Find the earliest troponin elevation across all patient visits."""
        troponin_tests = evidence.get("troponin", {}).get("troponin_tests", [])

        if not troponin_tests:
            return None

        # Sort by timestamp and find first elevated test
        elevated_tests = [
            test for test in troponin_tests if test.get("above_threshold")
        ]
        if not elevated_tests:
            return None

        # Sort by timestamp
        elevated_tests.sort(key=lambda x: x.get("timestamp", ""))
        earliest_test = elevated_tests[0]

        timestamp = earliest_test.get("timestamp")
        if timestamp:
            try:
                parsed_date = date_parse(timestamp)
                return parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Error parsing troponin timestamp {timestamp}: {e}")

        return None

    def _find_earliest_historical_symptoms(
        self, evidence: Dict[str, Any]
    ) -> Optional[str]:
        """Find the earliest symptom onset across all patient visits."""
        symptoms = evidence.get("clinical", {}).get("symptoms", [])

        if not symptoms:
            return None

        earliest_date = None

        for symptom in symptoms:
            onset_time = symptom.get("onset_time")
            if onset_time and onset_time != "unspecified":
                try:
                    # Try to parse relative dates like "yesterday morning", "2 days ago"
                    parsed_date = self._parse_relative_date(onset_time)
                    if parsed_date:
                        if earliest_date is None or parsed_date < earliest_date:
                            earliest_date = parsed_date
                except Exception as e:
                    logger.debug(
                        f"Could not parse symptom onset time '{onset_time}': {e}"
                    )

        if earliest_date:
            return earliest_date.strftime("%Y-%m-%d")

        return None

    def _parse_relative_date(self, relative_date_str: str) -> Optional[Any]:
        """Parse relative date strings like 'yesterday morning', '2 days ago'."""
        # This would be enhanced with more sophisticated date parsing
        # For now, return None to indicate unparseable
        return None

    def _llm_analyze_historical_onset(
        self, patient_id: str, visit_history: List[Dict], evidence: Dict[str, Any]
    ) -> Optional[str]:
        """Use LLM to analyze complete patient history for MI onset date."""
        if not self.llm_client:
            return None

        try:
            # Compile comprehensive clinical context
            clinical_context = self._compile_historical_context(visit_history, evidence)

            # Create LLM prompt for temporal analysis
            prompt = self._create_onset_analysis_prompt(patient_id, clinical_context)

            # Get LLM analysis
            response = self.llm_client.call_llm(prompt)

            if response and "onset_date" in response:
                return response["onset_date"]

        except Exception as e:
            logger.error(
                f"LLM onset date analysis failed for patient {patient_id}: {e}"
            )

        return None

    def _compile_historical_context(
        self, visit_history: List[Dict], evidence: Dict[str, Any]
    ) -> str:
        """Compile comprehensive clinical context from all visits."""
        context_parts = []

        # Add visit chronology
        context_parts.append("PATIENT VISIT CHRONOLOGY:")
        for i, visit in enumerate(visit_history, 1):
            context_parts.append(
                f"Visit {i} ({visit['chartdate']}): {visit['hadm_id']}"
            )

        # Add clinical evidence summary
        context_parts.append("\nCLINICAL EVIDENCE SUMMARY:")

        # Troponin data
        troponin_tests = evidence.get("troponin", {}).get("troponin_tests", [])
        if troponin_tests:
            context_parts.append(f"Troponin tests: {len(troponin_tests)} total")
            elevated_count = sum(
                1 for test in troponin_tests if test.get("above_threshold")
            )
            context_parts.append(f"Elevated troponin tests: {elevated_count}")

        # Symptoms
        symptoms = evidence.get("clinical", {}).get("symptoms", [])
        if symptoms:
            context_parts.append(f"Symptoms documented: {len(symptoms)}")
            for symptom in symptoms[:5]:  # Show first 5 symptoms
                context_parts.append(
                    f"  - {symptom.get('symptom', 'Unknown')}: {symptom.get('onset_time', 'timing unspecified')}"
                )

        # Add clinical notes excerpts
        context_parts.append("\nCLINICAL NOTES EXCERPTS:")
        for visit in visit_history:
            # Add first 500 characters of each visit's notes
            text_excerpt = (
                visit["text"][:500] + "..."
                if len(visit["text"]) > 500
                else visit["text"]
            )
            context_parts.append(f"\nVisit {visit['hadm_id']} ({visit['chartdate']}):")
            context_parts.append(text_excerpt)

        return "\n".join(context_parts)

    def _create_onset_analysis_prompt(
        self, patient_id: str, clinical_context: str
    ) -> str:
        """Create LLM prompt for MI onset date analysis."""
        return f"""
Analyze the complete clinical history for patient {patient_id} to determine the most accurate Myocardial Infarction onset date.

CLINICAL CONTEXT:
{clinical_context}

TASK:
Based on the chronological clinical evidence, determine the most likely MI onset date. Consider:
1. First mention of cardiac symptoms (chest pain, shortness of breath, etc.)
2. First elevated cardiac biomarkers (troponin)
3. Clinical timeline and progression
4. Any explicit mentions of symptom onset timing

RESPONSE FORMAT:
{{
    "onset_date": "YYYY-MM-DD or null if undetermined",
    "rationale": "Brief explanation of the reasoning",
    "confidence": "high/medium/low"
}}

Focus on finding the earliest credible evidence of MI onset, not just the diagnosis date.
"""
