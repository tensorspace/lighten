"""Troponin analyzer for detecting myocardial infarction patterns."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
from dateutil.parser import parse as date_parse

from .unit_converter import (
    convert_troponin_units,
    is_above_troponin_threshold,
    compare_troponin_values,
)
from ..llm_client import LightenLLMClient

logger = logging.getLogger(__name__)


class TroponinAnalyzer:
    """Analyzes troponin data to determine if it meets MI criteria."""

    # Diagnostic threshold for Troponin T in ng/mL, 5x for high-value exception
    TROPONIN_THRESHOLD = 0.014
    HIGH_VALUE_THRESHOLD = TROPONIN_THRESHOLD * 5

    def __init__(self, time_window_hours: int = 72):
        """Initialize the TroponinAnalyzer.

        Args:
            time_window_hours: The time window in hours for rise/fall analysis.
        """
        self.time_window = timedelta(hours=time_window_hours)

    def process_troponin_data(
        self, troponin_df: pd.DataFrame, patient_id: str
    ) -> Dict[str, Any]:
        """Processes a patient's troponin history to check for MI criteria.

        Args:
            troponin_df: DataFrame containing the patient's troponin history.
            patient_id: The ID of the patient.

        Returns:
            Dictionary containing troponin analysis results.
        """
        logger.info(
            f"[{patient_id}] Starting troponin analysis for {len(troponin_df)} records."
        )
        evidence = self._get_evidence_base()

        if troponin_df.empty:
            logger.warning(f"[{patient_id}] No troponin data provided.")
            evidence["troponin_available"] = False
            return evidence

        # Process troponin values (unit conversion, thresholding)
        processed_tests = self._process_troponin_tests(troponin_df, patient_id)
        max_value = max(
            (
                t["converted_value"]
                for t in processed_tests
                if t.get("converted_value") is not None
            ),
            default=None,
        )

        logger.info(
            f"[{patient_id}] Max troponin value: {max_value if max_value is not None else 'N/A'}"
        )

        # Check for MI criteria
        criteria_met, criteria_details = self._check_mi_criteria(
            processed_tests, patient_id
        )
        logger.info(
            f"[{patient_id}] Troponin criteria result: {'MET' if criteria_met else 'NOT MET'}. Details: {criteria_details}"
        )

        evidence.update(
            {
                "troponin_available": True,
                "troponin_tests": processed_tests,
                "max_troponin": max_value,
                "mi_criteria_met": criteria_met,
                "criteria_details": criteria_details,
            }
        )

        return evidence

    def _get_evidence_base(self) -> Dict[str, Any]:
        """Return the base structure for evidence."""
        return {
            "troponin_available": False,
            "troponin_tests": [],
            "max_troponin": None,
            "mi_criteria_met": False,
            "criteria_details": {},
        }

    def _process_troponin_tests(
        self, troponin_df: pd.DataFrame, patient_id: str
    ) -> List[Dict[str, Any]]:
        """Standardize and process a DataFrame of troponin tests."""
        processed_tests = []
        if troponin_df.empty:
            return processed_tests

        logger.debug(
            f"[{patient_id}] Standardizing {len(troponin_df)} troponin records."
        )

        # Ensure 'charttime' is datetime
        if "charttime" in troponin_df.columns:
            troponin_df["charttime"] = pd.to_datetime(
                troponin_df["charttime"], errors="coerce"
            )

        for _, row in troponin_df.iterrows():
            original_value = row.get("valuenum")
            original_unit = row.get("valueuom")

            if pd.isna(original_value):
                continue

            # Perform unit conversion
            converted_value, converted_unit, conversion_details = (
                convert_troponin_units(original_value, original_unit)
            )

            if conversion_details:
                logger.debug(
                    f"[{patient_id}] Unit conversion performed: {conversion_details}"
                )

            # Check if value is above diagnostic threshold
            is_above, threshold_details = is_above_troponin_threshold(
                converted_value, converted_unit, self.TROPONIN_THRESHOLD
            )

            processed_tests.append(
                {
                    "charttime": row.get("charttime"),
                    "original_value": original_value,
                    "original_unit": original_unit,
                    "converted_value": converted_value,
                    "converted_unit": converted_unit,
                    "above_threshold": is_above,
                    "threshold_details": threshold_details,
                }
            )

        # Sort by time
        processed_tests.sort(key=lambda x: x["charttime"])
        return processed_tests

    def _check_mi_criteria(
        self, troponin_values: List[Dict], patient_id: str
    ) -> Tuple[bool, Dict]:
        """Check if troponin values meet the 4th Universal Definition of MI.

        Criteria:
        A) A rise and/or fall of cTn with at least one value above the 99th percentile URL.
        B) A single troponin value >5x the diagnostic threshold.
        """
        if not troponin_values:
            return False, {"reason": "No troponin values available"}

        # Criterion B: Check for a single markedly elevated value (>5x threshold)
        for t in troponin_values:
            if (
                t["converted_value"]
                and t["converted_value"] > self.HIGH_VALUE_THRESHOLD
            ):
                details = {
                    "reason": "Single troponin value >5x diagnostic threshold",
                    "value": t["converted_value"],
                    "threshold": self.HIGH_VALUE_THRESHOLD,
                }
                logger.info(f"[{patient_id}] MI Criterion B met: {details}")
                return True, details

        # Criterion A: Detect a significant rise/fall pattern
        pattern, pattern_details = self._detect_rise_fall_pattern(troponin_values)
        if pattern:
            logger.info(f"[{patient_id}] MI Criterion A met: {pattern_details}")
            return True, pattern_details

        logger.info(f"[{patient_id}] No MI-defining troponin pattern found.")
        return False, {
            "reason": "No significant rise/fall or markedly elevated value detected"
        }

    def _detect_rise_fall_pattern(
        self, troponin_values: List[Dict]
    ) -> Tuple[bool, Dict]:
        """Detects a rise and/or fall pattern in a series of troponin values."""
        if len(troponin_values) < 2:
            return False, {
                "reason": "Insufficient data for pattern analysis (<2 values)"
            }

        for i in range(len(troponin_values) - 1):
            for j in range(i + 1, len(troponin_values)):
                val1 = troponin_values[i]
                val2 = troponin_values[j]

                # Ensure at least one value is above the threshold
                if not (val1["above_threshold"] or val2["above_threshold"]):
                    continue

                # Ensure values are within the time window
                time_diff = abs(val2["charttime"] - val1["charttime"])
                if time_diff > self.time_window:
                    continue

                # Validate the pattern's significance (rise/fall percentage)
                is_significant, details = self._validate_pattern_significance(
                    val1, val2
                )
                if is_significant:
                    return True, details

        return False, {"reason": "No significant dynamic change found"}

    def _validate_pattern_significance(
        self, val1: Dict, val2: Dict
    ) -> Tuple[bool, Dict]:
        """Validates if the change between two troponin values is significant.

        - Rise: >= 50% increase
        - Fall: >= 25% decrease
        """
        v1 = val1["converted_value"]
        v2 = val2["converted_value"]

        if v1 is None or v2 is None or v1 == 0:
            return False, {}

        change_percent = ((v2 - v1) / v1) * 100

        details = {
            "start_value": v1,
            "end_value": v2,
            "start_time": val1["charttime"].isoformat(),
            "end_time": val2["charttime"].isoformat(),
            "change_percent": round(change_percent, 2),
        }

        # Check for significant rise (>= 50%)
        if change_percent >= 50:
            details["pattern"] = "rise"
            return True, details

        # Check for significant fall (>= 25% decrease, so change is <= -25)
        if change_percent <= -25:
            details["pattern"] = "fall"
            return True, details

        return False, {}
