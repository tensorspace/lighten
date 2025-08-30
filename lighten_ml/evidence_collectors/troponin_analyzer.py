"""Troponin analyzer for detecting myocardial infarction patterns."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

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
                # Try both 'valuenum' and 'value' column names for compatibility
                value = None
                if "valuenum" in test and test["valuenum"] is not None:
                    value = float(test["valuenum"])
                elif "value" in test and test["value"] is not None:
                    value = float(test["value"])
                else:
                    logger.warning(
                        f"No numeric value found in test record: {test.to_dict()}"
                    )
                    continue

                # Get units information
                unit = test.get("valueuom", "")

                # Use charttime as primary time reference (clinical time when measurement was taken)
                # Fall back to storetime if charttime is not available
                timestamp = None
                if "charttime" in test and test["charttime"] is not None:
                    timestamp = (
                        test["charttime"]
                        if hasattr(test["charttime"], "year")
                        else date_parse(str(test["charttime"]))
                    )
                elif "storetime" in test and test["storetime"] is not None:
                    timestamp = (
                        test["storetime"]
                        if hasattr(test["storetime"], "year")
                        else date_parse(str(test["storetime"]))
                    )
                    logger.warning(
                        f"Using storetime as fallback for troponin test (charttime not available)"
                    )
                else:
                    logger.warning(f"No valid timestamp found in test record")

                # Convert units if necessary (troponin threshold is in ng/mL)
                logger.info(f"[TEST] TROPONIN TEST PROCESSING: Test {len(processed)+1}")
                logger.info(f"[TEST] TROPONIN TEST: Raw value = {value} {unit}")
                logger.info(f"[TEST] TROPONIN TEST: Timestamp = {timestamp}")

                converted_value, final_unit = convert_troponin_units(value, unit)

                if converted_value != value:
                    logger.info(f"[CONVERT] TROPONIN TEST: Unit conversion applied")
                    logger.info(
                        f"[CONVERT] TROPONIN TEST: {value} {unit} -> {converted_value} {final_unit}"
                    )
                    conversion_factor = converted_value / value if value != 0 else 1
                    logger.info(
                        f"[CONVERT] TROPONIN TEST: Conversion factor = {conversion_factor:.6f}"
                    )
                else:
                    logger.info(f"[PASS] TROPONIN TEST: No unit conversion needed")

                # Track maximum value (use converted value for comparison)
                if converted_value > max_value:
                    max_value = converted_value
                    logger.info(
                        f"[MAX] TROPONIN TEST: New maximum value = {max_value:.6f} ng/mL"
                    )

                # Use unit-aware threshold comparison
                logger.info(f"[THRESHOLD] TROPONIN TEST: Performing threshold comparison...")
                threshold_result = is_above_troponin_threshold(
                    value, unit, self.TROPONIN_THRESHOLD
                )

                # Log threshold comparison result
                if threshold_result["above_threshold"]:
                    logger.info(
                        f"[PASS] TROPONIN TEST: ABOVE THRESHOLD - {threshold_result['converted_value']} > {self.TROPONIN_THRESHOLD} ng/mL"
                    )
                    logger.info(
                        f"[PASS] TROPONIN TEST: Fold change = {threshold_result['fold_change']:.3f}x"
                    )
                else:
                    logger.info(
                        f"[FAIL] TROPONIN TEST: BELOW THRESHOLD - {threshold_result['converted_value']} <= {self.TROPONIN_THRESHOLD} ng/mL"
                    )

                processed.append(
                    {
                        "value": converted_value,  # Use converted value
                        "original_value": value,  # Keep original for reference
                        "unit": final_unit,  # Final unit after conversion
                        "original_unit": unit,  # Original unit for reference
                        "timestamp": timestamp,
                        "above_threshold": threshold_result["above_threshold"],
                        "threshold_analysis": threshold_result,  # Full unit-aware analysis
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

        # DETAILED DEBUGGING FOR TROPONIN ANALYSIS
        logger.info(f"TROPONIN DEBUG - Total values to analyze: {len(troponin_values)}")
        logger.info(f"TROPONIN DEBUG - Threshold: {self.TROPONIN_THRESHOLD} ng/mL")
        logger.info(f"TROPONIN DEBUG - 5x Threshold: {self.TROPONIN_THRESHOLD * 5} ng/mL")

        # Log first few troponin values for debugging
        for i, t in enumerate(troponin_values[:5]):
            logger.info(
                f"TROPONIN DEBUG - Value {i+1}: {t.get('value', 'N/A')} ng/mL, above_threshold: {t.get('above_threshold', 'N/A')}"
            )

        # Check for single elevated value scenarios FIRST (per clinical guideline)
        single_elevated_result = self._check_single_elevated_scenarios(troponin_values)
        if single_elevated_result["met"]:
            logger.info(f"TROPONIN DECISION - Single elevated scenario met: {single_elevated_result['type']}")
            return True, {
                "criteria": "Single elevated value scenario",
                "details": single_elevated_result,
                "decision_basis": single_elevated_result["reason"],
            }

        # Standard analysis for multiple values
        has_one_elevated = any(t["above_threshold"] for t in troponin_values)
        elevated_count = sum(1 for t in troponin_values if t["above_threshold"])
        max_value = max(t.get('value', 0) for t in troponin_values)

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
                    logger.info(f"TROPONIN RISE PATTERN - Case 2: Significant increase {prev_val} → {curr_val} ng/mL ({increase_pct:.1f}% increase)")
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
                logger.info(f"TROPONIN FALL PATTERN - Case 1: Peak decline {prev_val} → {curr_val} ng/mL ({decrease_pct:.1f}% decrease)")
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
            elif prev_val > self.TROPONIN_THRESHOLD and curr_val < prev_val and curr_val <= 0.75 * prev_val:
                decrease_pct = ((prev_val - curr_val) / prev_val) * 100
                logger.info(f"TROPONIN FALL PATTERN - Case 2: Elevated baseline decline {prev_val} → {curr_val} ng/mL ({decrease_pct:.1f}% decrease)")
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
            value = val_dict.get('value', 0)
            
            # Scenario 1: Single troponin >5x threshold
            if value > five_x_threshold:
                logger.info(f"TROPONIN SINGLE ELEVATED - Value {i+1}: {value} ng/mL > {five_x_threshold} ng/mL (5x threshold)")
                result = {
                    "met": True,
                    "type": "single_5x_threshold",
                    "reason": f"Single troponin value >5x threshold: {value} ng/mL > {five_x_threshold} ng/mL",
                    "value": value,
                    "threshold_multiple": value / self.TROPONIN_THRESHOLD,
                    "index": i
                }
                return result
        
        # Scenario 2: Clinical presentation + single troponin
        # Note: This requires clinical context integration (future enhancement)
        if len(values) == 1 and values[0].get('above_threshold', False):
            single_value = values[0].get('value', 0)
            logger.info(f"TROPONIN SINGLE ELEVATED - Single troponin above threshold: {single_value} ng/mL")
            logger.info(f"TROPONIN SINGLE ELEVATED - Clinical context required for single troponin exception")
            # For now, we note this but don't automatically qualify it
            # This will be handled by clinical context integration
        
        return result

    def _validate_pattern_significance(self, prev_val: float, curr_val: float, pattern_type: str) -> bool:
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
                    logger.info(f"TROPONIN PATTERN REJECTED - Rise pattern insufficient: {increase_pct:.1f}% < 50% required")
                    return False
        
        elif pattern_type == "fall":
            # For fall: require ≥25% decrease
            decrease_pct = ((prev_val - curr_val) / prev_val) * 100
            if decrease_pct < 25:
                logger.info(f"TROPONIN PATTERN REJECTED - Fall pattern insufficient: {decrease_pct:.1f}% < 25% required")
                return False
        
        return True
