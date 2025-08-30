"""Troponin analyzer for detecting myocardial infarction patterns."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
from dateutil.parser import parse as date_parse

from .base_evidence_collector import BaseEvidenceCollector
from .unit_converter import (
    convert_troponin_units,
    is_above_troponin_threshold,
    compare_troponin_values,
)
from ..llm_client import LightenLLMClient
from ..resolvers.mi_onset_date_resolver import MIOnsetDateResolver

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
        self.llm_client = LightenLLMClient()
        self.onset_date_resolver = MIOnsetDateResolver()

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect and analyze troponin evidence for a specific admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing troponin analysis results
        """
        # Use patient_id for logging instead of hadm_id (which can be null)
        logger.info(f"[PATIENT_{patient_id}] === TROPONIN ANALYSIS START ===")
        logger.info(
            f"[PATIENT_{patient_id}] Analyzing admission: {hadm_id or 'UNKNOWN'}"
        )
        logger.debug(
            f"[DEBUG] Patient {patient_id} - Single admission troponin analysis mode"
        )
        evidence = self._get_evidence_base()

        # Get all troponin tests for the patient's admission
        logger.debug(
            f"[DEBUG] Patient {patient_id} - Querying lab data for admission {hadm_id}"
        )
        troponin_tests = self.lab_data_loader.get_troponin_tests(patient_id, hadm_id)
        logger.info(
            f"[PATIENT_{patient_id}] Found {len(troponin_tests)} troponin test records for admission {hadm_id or 'UNKNOWN'}"
        )

        if not troponin_tests.empty:
            # Log test distribution
            test_dates = (
                troponin_tests["charttime"].dt.date.value_counts().sort_index()
                if "charttime" in troponin_tests.columns
                else None
            )
            if test_dates is not None and len(test_dates) > 0:
                logger.info(
                    f"[PATIENT_{patient_id}] Tests span {len(test_dates)} days: {test_dates.index[0]} to {test_dates.index[-1]}"
                )
                logger.debug(
                    f"[DEBUG] Patient {patient_id} - Daily test distribution: {dict(test_dates)}"
                )

        if troponin_tests.empty:
            logger.warning(
                f"[PATIENT_{patient_id}] No troponin tests found for admission {hadm_id or 'UNKNOWN'}"
            )
            evidence["troponin_available"] = False
            return evidence

        # Process troponin values
        processed = self._process_troponin_tests(troponin_tests, patient_id)
        logger.info(
            f"[PATIENT_{patient_id}] Processed troponin values: max={processed.get('max_value', 'N/A')}, count={len(processed.get('values', []))}"
        )

        # Check for MI criteria
        criteria_met, criteria_details = self._check_mi_criteria(
            processed["values"], patient_id
        )
        logger.info(
            f"[PATIENT_{patient_id}] *** TROPONIN CRITERIA RESULT: {'MET' if criteria_met else 'NOT MET'} ***"
        )
        logger.info(
            f"[PATIENT_{patient_id}] Troponin criteria details: {criteria_details}"
        )

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

    def process_troponin_data(self, patient_id: str, troponin_data: pd.DataFrame) -> Dict[str, Any]:
        """Processes a patient's complete troponin history data.

        Args:
            patient_id: The ID of the patient.
            troponin_data: DataFrame containing the patient's troponin history.

        Returns:
            Dictionary containing troponin analysis results.
        """
        logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] Processing {len(troponin_data)} troponin tests...")
        evidence = self._get_evidence_base()

        if troponin_data.empty:
            logger.warning(f"[{patient_id}] [TROPONIN_ANALYZER] No troponin data provided.")
            evidence["troponin_available"] = False
            return evidence

        # Process troponin values (unit conversion, thresholding)
        processed_tests = self._process_troponin_tests(troponin_data, patient_id)
        max_value = max([t['converted_value'] for t in processed_tests if t.get('converted_value') is not None], default=None)

        logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] Max troponin value: {max_value if max_value is not None else 'N/A'}")

        evidence.update(
            {
                "troponin_available": True,
                "troponin_tests": processed_tests,
                "max_troponin": max_value,
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
            "sources": [],
        }

    def _process_troponin_tests(
        self, troponin_df: pd.DataFrame, patient_id: str
    ) -> List[Dict[str, Any]]:
        """Standardize and process a DataFrame of troponin tests."""
        processed_tests = []
        if troponin_df.empty:
            return processed_tests

        logger.debug(f"[{patient_id}] [TROPONIN_ANALYZER] Standardizing {len(troponin_df)} troponin records.")

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
            converted_value, converted_unit, conversion_details = convert_troponin_units(
                original_value, original_unit
            )

            if conversion_details:
                logger.debug(f"[{patient_id}] [TROPONIN_ANALYZER] {conversion_details}")

            # Check if value is above threshold
            above_threshold, threshold_details = is_above_troponin_threshold(
                converted_value, self.TROPONIN_THRESHOLD
            )

            test_record = {
                "charttime": row.get("charttime"),
                "original_value": original_value,
                "original_unit": original_unit,
                "converted_value": converted_value,
                "converted_unit": converted_unit,
                "above_threshold": above_threshold,
                "threshold_details": threshold_details,
            }
            processed_tests.append(test_record)

        # Sort tests by time for chronological analysis
        processed_tests.sort(key=lambda x: x["charttime"] or datetime.min)
        logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] Standardized {len(processed_tests)} valid troponin records.")
        return processed_tests

    def _check_mi_criteria(
        self, processed_tests: List[Dict[str, Any]], patient_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check for MI criteria (rise/fall and value above threshold)."""
        details = {"rise_fall_pattern": False, "above_threshold": False}

        if not processed_tests or len(processed_tests) < 2:
            logger.debug(f"[{patient_id}] [TROPONIN_ANALYZER] MI criteria check requires at least 2 tests.")
            return False, details

        # Check for at least one value above the 99th percentile URL
        high_values = [t for t in processed_tests if t.get("above_threshold")]
        if not high_values:
            logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] No troponin values above threshold found.")
            return False, details

        details["above_threshold"] = True
        logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] Found {len(high_values)} value(s) above threshold.")

        # Check for a significant rise and/or fall pattern
        has_rise_fall = self._detect_rise_fall_pattern(processed_tests, patient_id)
        if not has_rise_fall:
            logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] No significant rise/fall pattern detected.")
            return False, details

        details["rise_fall_pattern"] = True
        logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] Significant rise/fall pattern DETECTED.")

        return True, details

    def _detect_rise_fall_pattern(
        self, processed_tests: List[Dict[str, Any]], patient_id: str
    ) -> bool:
        """Detects a rise and/or fall pattern in a series of troponin tests."""
        if len(processed_tests) < 2:
            return False

        for i in range(len(processed_tests) - 1):
            for j in range(i + 1, len(processed_tests)):
                t1 = processed_tests[i]
                t2 = processed_tests[j]

                # Ensure time difference is within the analysis window
                time_diff = abs(t2["charttime"] - t1["charttime"])
                if time_diff > self.time_window:
                    continue

                comparison = compare_troponin_values(
                    t1.get("converted_value"), t2.get("converted_value")
                )

                # Check for significant change (e.g., >20% change)
                if comparison["significant_change"]:
                    logger.debug(
                        f"[{patient_id}] [TROPONIN_ANALYZER] Significant change found: "
                        f"{t1['converted_value']:.4f} -> {t2['converted_value']:.4f} "
                        f"over {time_diff}"
                    )
                    return True

        return False

    def _check_single_elevated_scenarios(self, values: List[Dict]) -> Dict:
        """Check for single elevated value scenarios per clinical guideline.

        Scenarios that meet criteria without rise/fall pattern:
        (1) Single troponin >5x threshold (>0.07 ng/mL)
        (2) Clinical presentation + single troponin (requires clinical context)
        """
        result = {"met": False, "type": None, "reason": None, "value": None}

        # Calculate 5x threshold (0.014 * 5 = 0.07 ng/mL)
        five_x_threshold = self.TROPONIN_THRESHOLD * 5

        # Check each value for >5x threshold scenario
        for i, val_dict in enumerate(values):
            value = val_dict.get("value", 0)

            # Scenario 1: Single troponin >5x threshold
            if value > five_x_threshold:
                logger.info(
                    f"TROPONIN SINGLE ELEVATED - Value {i+1}: {value} ng/mL > {five_x_threshold} ng/mL (5x threshold)"
                )
                result = {
                    "met": True,
                    "type": "single_5x_threshold",
                    "reason": f"Single troponin value >5x threshold: {value} ng/mL > {five_x_threshold} ng/mL",
                    "value": value,
                    "threshold_multiple": value / self.TROPONIN_THRESHOLD,
                    "index": i,
                }
                return result

        # Scenario 2: Clinical presentation + single troponin
        # Note: This requires clinical context integration (future enhancement)
        if len(values) == 1 and values[0].get("above_threshold", False):
            single_value = values[0].get("value", 0)
            logger.info(
                f"TROPONIN SINGLE ELEVATED - Single troponin above threshold: {single_value} ng/mL"
            )
            logger.info(
                f"TROPONIN SINGLE ELEVATED - Clinical context required for single troponin exception"
            )
            # For now, we note this but don't automatically qualify it
            # This will be handled by clinical context integration

        return result

    def _validate_pattern_significance(
        self, prev_val: float, curr_val: float, pattern_type: str
    ) -> bool:
        """Validate that patterns meet minimum significance thresholds per clinical guideline.

        Args:
            prev_val: Previous troponin value
            curr_val: Current troponin value
            pattern_type: Type of pattern ('rise' or 'fall')

        Returns:
            True if pattern meets significance threshold, False otherwise
        """
        if prev_val <= 0:  # Avoid division by zero
            return False

        if pattern_type == "rise":
            # For rise: require ≥50% increase when baseline is above threshold
            if prev_val > self.TROPONIN_THRESHOLD:
                increase_pct = ((curr_val - prev_val) / prev_val) * 100
                if increase_pct < 50:
                    logger.info(
                        f"TROPONIN PATTERN REJECTED - Rise pattern insufficient: {increase_pct:.1f}% < 50% required"
                    )
                    return False

        elif pattern_type == "fall":
            # For fall: require ≥25% decrease
            decrease_pct = ((prev_val - curr_val) / prev_val) * 100
            if decrease_pct < 25:
                logger.info(
                    f"TROPONIN PATTERN REJECTED - Fall pattern insufficient: {decrease_pct:.1f}% < 25% required"
                )
                return False

        return True

    def _analyze_temporal_pattern(self, values: List[Dict], patient_id: str) -> str:
        """Analyze temporal patterns in troponin values.

        Args:
            values: List of processed troponin values
            patient_id: Patient ID for logging

        Returns:
            String describing the temporal pattern
        """
        if len(values) < 2:
            return "Insufficient data for temporal analysis"

        # Sort by timestamp
        sorted_values = sorted(values, key=lambda x: x.get("timestamp", ""))

        # Analyze trend
        elevated_values = [v for v in sorted_values if v.get("above_threshold", False)]

        if not elevated_values:
            return "No elevated values detected"
        elif len(elevated_values) == 1:
            return "Single elevated value"
        else:
            # Check for rise/fall pattern
            if len(elevated_values) >= 2:
                return "Multiple elevated values - potential dynamic pattern"
            else:
                return "Isolated elevation"

    def _classify_historical_pattern(self, values: List[Dict], patient_id: str) -> str:
        """Classify the overall historical pattern of troponin values.

        Args:
            values: List of processed troponin values
            patient_id: Patient ID for logging

        Returns:
            String classification of the historical pattern
        """
        if not values:
            return "No troponin data"

        elevated_count = sum(1 for v in values if v.get("above_threshold", False))
        total_count = len(values)
        elevation_rate = (elevated_count / total_count) * 100

        if elevation_rate == 0:
            return "No elevations detected"
        elif elevation_rate < 25:
            return "Occasional elevations"
        elif elevation_rate < 75:
            return "Frequent elevations"
        else:
            return "Persistent elevations"

    def _check_mi_criteria(self, troponin_values: List[Dict]) -> Tuple[bool, Dict]:
        """Check if troponin values meet MI criteria.

        Args:
            troponin_values: List of processed troponin values with timestamps

        Returns:
            Tuple of (criteria_met, criteria_details)
        """
        if not troponin_values:
            return False, {"reason": "No troponin values available"}

        # DETAILED DEBUGGING FOR TROPONIN ANALYSIS
        logger.info(f"TROPONIN DEBUG - Total values to analyze: {len(troponin_values)}")
        logger.info(f"TROPONIN DEBUG - Threshold: {self.TROPONIN_THRESHOLD} ng/mL")
        logger.info(
            f"TROPONIN DEBUG - 5x Threshold: {self.TROPONIN_THRESHOLD * 5} ng/mL"
        )

        # Log first few troponin values for debugging
        for i, t in enumerate(troponin_values[:5]):
            logger.info(
                f"TROPONIN DEBUG - Value {i+1}: {t.get('value', 'N/A')} ng/mL, above_threshold: {t.get('above_threshold', 'N/A')}"
            )

        # Check for single elevated value scenarios FIRST (per clinical guideline)
        single_elevated_result = self._check_single_elevated_scenarios(troponin_values)
        if single_elevated_result["met"]:
            logger.info(
                f"TROPONIN DECISION - Single elevated scenario met: {single_elevated_result['type']}"
            )
            return True, {
                "criteria": "Single elevated value scenario",
                "details": single_elevated_result,
                "decision_basis": single_elevated_result["reason"],
            }

        # Standard analysis for multiple values
        has_one_elevated = any(t["above_threshold"] for t in troponin_values)
        elevated_count = sum(1 for t in troponin_values if t["above_threshold"])
        max_value = max(t.get("value", 0) for t in troponin_values)

        logger.info(f"TROPONIN DEBUG - Has one elevated: {has_one_elevated}")
        logger.info(f"TROPONIN DEBUG - Total elevated values: {elevated_count}")
        logger.info(f"TROPONIN DEBUG - Max value in dataset: {max_value} ng/mL")

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
        logger.info(f"TROPONIN ANALYSIS - Checking for rise patterns...")
        rise_result = self._check_rise_pattern(troponin_values)
        if rise_result["met"]:
            logger.info(
                f"TROPONIN DECISION - Rise pattern detected: {rise_result['details']}"
            )
            return True, {
                "criteria": "Rise pattern detected",
                "details": rise_result,
                "decision_basis": "Dynamic rise pattern in troponin levels",
            }

        # Check for fall pattern
        logger.info(f"TROPONIN ANALYSIS - Checking for fall patterns...")
        fall_result = self._check_fall_pattern(troponin_values)
        if fall_result["met"]:
            logger.info(
                f"TROPONIN DECISION - Fall pattern detected: {fall_result['details']}"
            )
            return True, {
                "criteria": "Fall pattern detected",
                "details": fall_result,
                "decision_basis": "Dynamic fall pattern in troponin levels",
            }

        # If no pattern, but one value is elevated, report that.
        if has_one_elevated:
            logger.info(
                f"TROPONIN DECISION - Single elevated troponin found (no dynamic pattern)"
            )
            logger.info(
                f"TROPONIN DECISION - Elevated values count: {elevated_count} out of {len(troponin_values)}"
            )
            return True, {
                "criteria": "Single elevated troponin",
                "reason": "One or more troponin values were above the threshold, but no rise/fall pattern was confirmed.",
                "decision_basis": f"At least one troponin value above {self.TROPONIN_THRESHOLD} ng/mL threshold",
            }

        logger.warning(
            f"TROPONIN DECISION - No MI criteria met: max_value={max(t.get('value', 0) for t in troponin_values)}, threshold={self.TROPONIN_THRESHOLD}"
        )
        return False, {
            "reason": "No MI criteria met",
            "decision_basis": "No troponin values above threshold and no dynamic patterns detected",
        }

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
                # Validate pattern significance before accepting
                if self._validate_pattern_significance(prev_val, curr_val, "rise"):
                    increase_pct = ((curr_val - prev_val) / prev_val) * 100
                    logger.info(
                        f"TROPONIN RISE PATTERN - Case 2: Significant increase {prev_val} → {curr_val} ng/mL ({increase_pct:.1f}% increase)"
                    )
                    result["met"] = True
                    result["details"].append(
                        {
                            "type": "significant_increase",
                            "from": prev_val,
                            "to": curr_val,
                            "increase_pct": increase_pct,
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

            # Validate pattern significance before accepting
            if not self._validate_pattern_significance(prev_val, curr_val, "fall"):
                continue

            # Case 1: Peak above threshold with subsequent decline (≥25%)
            if prev_val > self.TROPONIN_THRESHOLD and curr_val <= 0.75 * prev_val:
                decrease_pct = ((prev_val - curr_val) / prev_val) * 100
                logger.info(
                    f"TROPONIN FALL PATTERN - Case 1: Peak decline {prev_val} → {curr_val} ng/mL ({decrease_pct:.1f}% decrease)"
                )
                result["met"] = True
                result["details"].append(
                    {
                        "type": "peak_decline",
                        "from": prev_val,
                        "to": curr_val,
                        "decrease_pct": decrease_pct,
                        "threshold_pct": 25,
                        "indices": (i - 1, i),
                    }
                )

            # Case 2: Declining from elevated baseline (≥25%)
            elif (
                prev_val > self.TROPONIN_THRESHOLD
                and curr_val < prev_val
                and curr_val <= 0.75 * prev_val
            ):
                decrease_pct = ((prev_val - curr_val) / prev_val) * 100
                logger.info(
                    f"TROPONIN FALL PATTERN - Case 2: Elevated baseline decline {prev_val} → {curr_val} ng/mL ({decrease_pct:.1f}% decrease)"
                )
                result["met"] = True
                result["details"].append(
                    {
                        "type": "elevated_baseline_decline",
                        "from": prev_val,
                        "to": curr_val,
                        "decrease_pct": decrease_pct,
                        "threshold_pct": 25,
                        "indices": (i - 1, i),
                    }
                )

        return result

    def _check_patient_mi_criteria(
        self, values: List[Dict], patient_id: str
    ) -> Tuple[bool, Dict]:
        """Check MI criteria using complete patient troponin history.

        Args:
            values: List of processed troponin values across all admissions
            patient_id: Patient ID for logging

        Returns:
            Tuple of (criteria_met: bool, criteria_details: dict)
        """
        logger.debug(
            f"[{patient_id}] [TROPONIN_ANALYZER] Evaluating MI criteria using {len(values)} troponin tests from complete patient history"
        )

        # Use existing criteria check but with enhanced patient-level logging
        criteria_met, criteria_details = self._check_mi_criteria(values, patient_id)

        # Add patient-level context to criteria details
        if criteria_met:
            logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] *** PATIENT-LEVEL MI CRITERIA MET ***")
            logger.info(
                f"[{patient_id}] [TROPONIN_ANALYZER] Criteria type: {criteria_details.get('type', 'Unknown')}"
            )
            logger.info(
                f"[{patient_id}] [TROPONIN_ANALYZER] Supporting evidence: {criteria_details.get('reason', 'Not specified')}"
            )
        else:
            logger.info(f"[{patient_id}] [TROPONIN_ANALYZER] Patient-level MI criteria NOT met")
            logger.debug(
                f"[{patient_id}] [TROPONIN_ANALYZER] Reason: {criteria_details.get('reason', 'Insufficient evidence')}"
            )

        # Enhance criteria details with patient-level context
        criteria_details["patient_level_analysis"] = True
        criteria_details["total_tests_analyzed"] = len(values)

        return criteria_met, criteria_details
