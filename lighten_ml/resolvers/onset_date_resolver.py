"""Resolver for determining the onset date of Myocardial Infarction."""

from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil.parser import parse as date_parse


class OnsetDateResolver:
    """Determines the MI onset date based on a hierarchy of evidence."""

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
        # 1. Symptom Onset Date (from clinical notes)
        symptom_onset = self._find_earliest_symptom_date(
            evidence.get("clinical", {}).get("symptoms", [])
        )
        if symptom_onset:
            return {
                "onset_date": symptom_onset.isoformat(),
                "rationale": "Symptom onset date",
            }

        # 2. First Abnormal ECG Date
        ecg_onset = self._find_earliest_ecg_date(
            evidence.get("ecg", {}).get("ecg_findings", [])
        )
        if ecg_onset:
            return {
                "onset_date": ecg_onset.isoformat(),
                "rationale": "First abnormal ECG date",
            }

        # 3. First Elevated Troponin Date
        troponin_onset = self._find_first_elevated_troponin_date(
            evidence.get("troponin", {}).get("troponin_tests", [])
        )
        if troponin_onset:
            return {
                "onset_date": troponin_onset.isoformat(),
                "rationale": "First elevated troponin date",
            }

        # 4. Clinical Recognition/Diagnosis Date
        diagnosis_date = self._find_earliest_diagnosis_date(
            evidence.get("clinical", {}).get("diagnoses", [])
        )
        if diagnosis_date:
            return {
                "onset_date": diagnosis_date.isoformat(),
                "rationale": "Clinical diagnosis date",
            }

        # 5. Hospital Presentation/Admission Date
        if admission_time:
            return {
                "onset_date": admission_time.isoformat(),
                "rationale": "Hospital admission date",
            }

        return {
            "onset_date": None,
            "rationale": "No definitive onset date could be determined",
        }

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
