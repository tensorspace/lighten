"""Troponin analyzer for detecting myocardial infarction patterns."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from dateutil.parser import parse as date_parse

from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class TroponinAnalyzer(BaseEvidenceCollector):
    """Analyzes troponin levels to detect myocardial infarction patterns."""

    # Diagnostic threshold for Troponin T in ng/mL
    TROPONIN_THRESHOLD = 0.014

    def __init__(self, lab_data_loader: Any, time_window_hours: int = 72):
        """Initialize the TroponinAnalyzer with a lab data loader.

        Args:
            lab_data_loader: Instance of LabDataLoader for accessing lab data
            time_window_hours: The time window in hours for rise/fall analysis.
        """
        super().__init__(lab_data_loader=lab_data_loader)
        self.time_window = timedelta(hours=time_window_hours)

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect and analyze troponin evidence for a specific admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing troponin analysis results
        """
        logger.info(f"[{hadm_id}] Starting troponin analysis for patient {patient_id}")
        evidence = self._get_evidence_base()

        # Get all troponin tests for the patient's admission
        troponin_tests = self.lab_data_loader.get_troponin_tests(patient_id, hadm_id)
        logger.info(f"[{hadm_id}] Found {len(troponin_tests)} troponin test records")

        if troponin_tests.empty:
            logger.warning(
                f"[{hadm_id}] No troponin tests found for patient {patient_id}"
            )
            evidence["troponin_available"] = False
            return evidence

        # Process troponin values
        processed = self._process_troponin_tests(troponin_tests)
        logger.info(
            f"[{hadm_id}] Processed troponin values: max={processed.get('max_value', 'N/A')}, count={len(processed.get('values', []))}"
        )

        # Check for MI criteria
        criteria_met, criteria_details = self._check_mi_criteria(processed["values"])
        logger.info(
            f"[{hadm_id}] *** TROPONIN CRITERIA RESULT: {'MET' if criteria_met else 'NOT MET'} ***"
        )
        logger.info(f"[{hadm_id}] Troponin criteria details: {criteria_details}")

        evidence.update(
            {
                "troponin_available": True,
                "troponin_tests": processed["values"],
                "max_troponin": processed["max_value"],
                "mi_criteria_met": criteria_met,
                "criteria_details": criteria_details,
                "sources": [
                    {
                        "type": "lab",
                        "description": "Troponin test results",
                        "count": len(processed["values"]),
                    }
                ],
            }
        )

        return evidence

    def _process_troponin_tests(self, troponin_tests: Any) -> Dict[str, Any]:
        """Process raw troponin test data.

        Args:
            troponin_tests: DataFrame containing troponin test results

        Returns:
            Dictionary containing processed troponin values and metadata
        """
        # Extract relevant columns and clean data
        processed = []
        max_value = 0.0

        for _, test in troponin_tests.iterrows():
            try:
                value = float(test["value"])
                timestamp_str = test.get("charttime", None)
                timestamp = date_parse(timestamp_str) if timestamp_str else None

                # Track maximum value
                if value > max_value:
                    max_value = value

                processed.append(
                    {
                        "value": value,
                        "timestamp": timestamp,
                        "above_threshold": value > self.TROPONIN_THRESHOLD,
                        "test_id": test.get("itemid"),
                        "test_name": test.get("label", "Troponin T"),
                    }
                )
            except (ValueError, TypeError):
                continue

        # Sort by timestamp
        processed.sort(key=lambda x: x.get("timestamp") or datetime.min)

        return {
            "values": processed,
            "max_value": max_value,
            "threshold": self.TROPONIN_THRESHOLD,
        }

    def _check_mi_criteria(self, troponin_values: List[Dict]) -> Tuple[bool, Dict]:
        """Check if troponin values meet MI criteria.

        Args:
            troponin_values: List of processed troponin values with timestamps

        Returns:
            Tuple of (criteria_met, criteria_details)
        """
        if not troponin_values:
            return False, {"reason": "No troponin values available"}

        # A single elevated value can be evidence, but rise/fall is stronger.
        # The final decision is made by the rule engine, which considers ischemia.
        has_one_elevated = any(t["above_threshold"] for t in troponin_values)

        # Need at least 2 values to check for rise/fall patterns
        if len(troponin_values) < 2:
            return has_one_elevated, {
                "reason": (
                    "Insufficient data for pattern analysis, but at least one value is elevated."
                    if has_one_elevated
                    else "Insufficient data for pattern analysis."
                ),
                "required_for_pattern": 2,
                "available": len(troponin_values),
            }

        # Check for rise pattern
        rise_result = self._check_rise_pattern(troponin_values)
        if rise_result["met"]:
            return True, {"criteria": "Rise pattern detected", "details": rise_result}

        # Check for fall pattern
        fall_result = self._check_fall_pattern(troponin_values)
        if fall_result["met"]:
            return True, {"criteria": "Fall pattern detected", "details": fall_result}

        # If no pattern, but one value is elevated, report that.
        if has_one_elevated:
            return True, {
                "criteria": "Single elevated troponin",
                "reason": "One or more troponin values were above the threshold, but no rise/fall pattern was confirmed.",
            }

        return False, {"reason": "No MI criteria met"}

    def _check_rise_pattern(self, values: List[Dict]) -> Dict:
        """Check for rise pattern in troponin values within the time window."""
        result = {"met": False, "pattern": "rise", "details": []}

        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]

            # Skip if values are missing or timestamps are invalid
            if (
                "value" not in prev
                or "value" not in curr
                or not prev["timestamp"]
                or not curr["timestamp"]
            ):
                continue

            # Check if within the time window
            if curr["timestamp"] - prev["timestamp"] > self.time_window:
                continue

            prev_val = prev["value"]
            curr_val = curr["value"]

            # Case 1: Baseline below threshold, subsequent above threshold
            if prev_val <= self.TROPONIN_THRESHOLD < curr_val:
                result["met"] = True
                result["details"].append(
                    {
                        "type": "below_to_above_threshold",
                        "from": prev_val,
                        "to": curr_val,
                        "threshold": self.TROPONIN_THRESHOLD,
                        "indices": (i - 1, i),
                    }
                )

            # Case 2: Significant increase from elevated baseline (≥50%)
            elif prev_val > self.TROPONIN_THRESHOLD and curr_val >= 1.5 * prev_val:
                result["met"] = True
                result["details"].append(
                    {
                        "type": "significant_increase",
                        "from": prev_val,
                        "to": curr_val,
                        "increase_pct": ((curr_val - prev_val) / prev_val) * 100,
                        "threshold_pct": 50,
                        "indices": (i - 1, i),
                    }
                )

        return result

    def _check_fall_pattern(self, values: List[Dict]) -> Dict:
        """Check for fall pattern in troponin values within the time window."""
        result = {"met": False, "pattern": "fall", "details": []}

        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]

            # Skip if values are missing or timestamps are invalid
            if (
                "value" not in prev
                or "value" not in curr
                or not prev["timestamp"]
                or not curr["timestamp"]
            ):
                continue

            # Check if within the time window
            if curr["timestamp"] - prev["timestamp"] > self.time_window:
                continue

            prev_val = prev["value"]
            curr_val = curr["value"]

            # Case 1: Peak above threshold with subsequent decline (≥25%)
            if prev_val > self.TROPONIN_THRESHOLD and curr_val <= 0.75 * prev_val:
                result["met"] = True
                result["details"].append(
                    {
                        "type": "significant_decline",
                        "from": prev_val,
                        "to": curr_val,
                        "decrease_pct": ((prev_val - curr_val) / prev_val) * 100,
                        "threshold_pct": 25,
                        "indices": (i - 1, i),
                    }
                )

        return result
