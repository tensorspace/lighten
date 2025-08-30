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
            f"[PATIENT_{patient_id}] 📊 Found {len(troponin_tests)} troponin test records for admission {hadm_id or 'UNKNOWN'}"
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
                    f"[PATIENT_{patient_id}] 📅 Tests span {len(test_dates)} days: {test_dates.index[0]} to {test_dates.index[-1]}"
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

    def analyze_patient_troponin_history(self, patient_id: str) -> Dict[str, Any]:
        """Analyze complete troponin history for a patient across all admissions.

        This method provides patient-level troponin analysis by aggregating
        all troponin tests across the patient's complete medical history.

        Args:
            patient_id: The ID of the patient

        Returns:
            Dictionary containing comprehensive patient-level troponin analysis
        """
        logger.info(
            f"[PATIENT_{patient_id}] === PATIENT-LEVEL TROPONIN ANALYSIS START ==="
        )
        logger.info(
            f"[PATIENT_{patient_id}] 🔍 Cross-admission troponin pattern analysis"
        )
        logger.debug(
            f"[DEBUG] Patient {patient_id} - Collecting complete troponin history across all admissions"
        )
        logger.info(
            f"[PATIENT_{patient_id}] 🎯 Analyzing temporal patterns for MI diagnosis"
        )

        evidence = self._get_evidence_base()

        # Get complete troponin history for the patient
        try:
            logger.debug(
                f"[DEBUG] Patient {patient_id} - Querying complete patient troponin history"
            )
            troponin_history = self.lab_data_loader.get_patient_troponin_history(
                patient_id
            )
            logger.info(
                f"[PATIENT_{patient_id}] 📊 Found {len(troponin_history)} total troponin tests across ALL admissions"
            )

            if not troponin_history.empty:
                # Log comprehensive history statistics
                unique_admissions = (
                    troponin_history["hadm_id"].nunique()
                    if "hadm_id" in troponin_history.columns
                    else 0
                )
                date_range = None
                if "charttime" in troponin_history.columns:
                    min_date = troponin_history["charttime"].min()
                    max_date = troponin_history["charttime"].max()
                    span_days = (
                        (max_date - min_date).days if min_date and max_date else 0
                    )
                    date_range = (
                        f"{min_date.date()} to {max_date.date()} ({span_days} days)"
                    )

                logger.info(
                    f"[PATIENT_{patient_id}] 🏥 Tests across {unique_admissions} admissions"
                )
                if date_range:
                    logger.info(f"[PATIENT_{patient_id}] 📅 Timeline: {date_range}")

                # Log value distribution
                if "valuenum" in troponin_history.columns:
                    values = troponin_history["valuenum"].dropna()
                    if len(values) > 0:
                        max_value = values.max()
                        min_value = values.min()
                        above_threshold = (values > self.troponin_threshold).sum()
                        logger.info(
                            f"[PATIENT_{patient_id}] 📈 Value range: {min_value:.3f} - {max_value:.3f} ng/mL"
                        )
                        logger.info(
                            f"[PATIENT_{patient_id}] 🎯 Tests above threshold (>{self.troponin_threshold}): {above_threshold}/{len(values)}"
                        )

            if troponin_history.empty:
                logger.warning(
                    f"[PATIENT_{patient_id}] No troponin tests found in patient's complete history"
                )
                evidence["troponin_available"] = False
                evidence["patient_level_analysis"] = {
                    "total_tests": 0,
                    "admissions_with_troponin": 0,
                    "date_range": None,
                    "historical_pattern": "No troponin data available",
                }
                return evidence

            # Process patient-level troponin data
            logger.info(
                f"[PATIENT_{patient_id}] 🔄 Processing cross-admission troponin patterns..."
            )
            processed = self._process_patient_troponin_history(
                troponin_history, patient_id
            )
            logger.info(
                f"[PATIENT_{patient_id}] 📋 Patient-level analysis summary: {processed['summary']}"
            )

            # Log detailed processing results
            if "patient_analysis" in processed:
                analysis = processed["patient_analysis"]
                logger.info(
                    f"[PATIENT_{patient_id}] 🏥 Admissions with troponin: {analysis.get('admissions_with_troponin', 0)}"
                )
                logger.info(
                    f"[PATIENT_{patient_id}] 📊 Total test count: {analysis.get('total_tests', 0)}"
                )
                if "date_range" in analysis and analysis["date_range"]:
                    logger.info(
                        f"[PATIENT_{patient_id}] 📅 Analysis period: {analysis['date_range']}"
                    )

            # Check for MI criteria using complete patient history
            logger.info(
                f"[PATIENT_{patient_id}] 🎯 Evaluating MI criteria using complete patient timeline..."
            )
            criteria_met, criteria_details = self._check_patient_mi_criteria(
                processed["values"], patient_id
            )
            logger.info(
                f"[PATIENT_{patient_id}] *** 🚨 PATIENT-LEVEL TROPONIN CRITERIA: {'✅ MET' if criteria_met else '❌ NOT MET'} ***"
            )

            # Log detailed criteria evaluation
            if criteria_details:
                logger.info(
                    f"[PATIENT_{patient_id}] 📝 Criteria details: {criteria_details.get('criteria_met', 'N/A')}"
                )
                if "rise_fall_pattern" in criteria_details:
                    logger.info(
                        f"[PATIENT_{patient_id}] 📈 Rise/fall pattern: {'✅ Detected' if criteria_details['rise_fall_pattern'] else '❌ Not detected'}"
                    )
                if "above_threshold" in criteria_details:
                    logger.info(
                        f"[PATIENT_{patient_id}] 🎯 Above threshold: {'✅ Yes' if criteria_details['above_threshold'] else '❌ No'}"
                    )

            evidence.update(
                {
                    "troponin_available": True,
                    "troponin_tests": processed["values"],
                    "max_troponin": processed["max_value"],
                    "mi_criteria_met": criteria_met,
                    "criteria_details": criteria_details,
                    "patient_level_analysis": processed["patient_analysis"],
                    "sources": [
                        {
                            "type": "lab_history",
                            "description": "Complete patient troponin history",
                            "count": len(processed["values"]),
                            "admissions": processed["patient_analysis"][
                                "admissions_with_troponin"
                            ],
                        }
                    ],
                }
            )

            return evidence

        except Exception as e:
            logger.error(
                f"[PATIENT_{patient_id}] ❌ ERROR in patient-level troponin analysis: {e}"
            )
            logger.error(
                f"[PATIENT_{patient_id}] 🚨 Cross-admission analysis failed - falling back to error state"
            )
            logger.debug(
                f"[DEBUG] Patient {patient_id} - Exception details", exc_info=True
            )
            evidence["troponin_available"] = False
            evidence["error"] = f"Patient-level analysis failed: {str(e)}"
            evidence["analysis_type"] = "failed_patient_level"
            return evidence

    def _process_troponin_tests(
        self, troponin_tests: Any, patient_id: str = None
    ) -> Dict[str, Any]:
        """Process raw troponin test data.

        Args:
            troponin_tests: DataFrame containing troponin test results
            patient_id: Patient ID for logging (optional, for backward compatibility)

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

                # Critical: Skip NaN values - they cannot be used for clinical analysis
                if (
                    pd.isna(value)
                    or not isinstance(value, (int, float))
                    or value != value
                ):  # NaN check
                    log_prefix = (
                        f"[PATIENT_{patient_id}]" if patient_id else "[TROPONIN]"
                    )
                    logger.warning(
                        f"{log_prefix} [SKIP] TROPONIN TEST: Invalid value (NaN) detected - skipping test"
                    )
                    logger.warning(
                        f"{log_prefix} [DATA_QUALITY] Raw value from database: {test.get('valuenum', test.get('value', 'N/A'))}"
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
                logger.info(
                    f"[THRESHOLD] TROPONIN TEST: Performing threshold comparison..."
                )
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

    def _process_patient_troponin_history(
        self, troponin_history: pd.DataFrame, patient_id: str
    ) -> Dict[str, Any]:
        """Process complete patient troponin history across all admissions.

        Args:
            troponin_history: DataFrame containing all troponin tests for the patient
            patient_id: Patient ID for logging

        Returns:
            Dictionary containing processed patient-level troponin analysis
        """
        logger.debug(
            f"[PATIENT_{patient_id}] Processing {len(troponin_history)} troponin tests from patient history"
        )

        # Process all troponin tests using existing method
        processed = self._process_troponin_tests(troponin_history, patient_id)

        # Add patient-level analysis metadata
        unique_admissions = (
            troponin_history["hadm_id"].nunique()
            if "hadm_id" in troponin_history.columns
            else 0
        )
        date_range_start = (
            troponin_history["charttime"].min()
            if "charttime" in troponin_history.columns
            else None
        )
        date_range_end = (
            troponin_history["charttime"].max()
            if "charttime" in troponin_history.columns
            else None
        )

        # Calculate elevated test statistics
        elevated_tests = [
            test for test in processed["values"] if test.get("above_threshold", False)
        ]
        elevated_count = len(elevated_tests)

        # Analyze temporal patterns
        temporal_pattern = self._analyze_temporal_pattern(
            processed["values"], patient_id
        )

        patient_analysis = {
            "total_tests": len(processed["values"]),
            "admissions_with_troponin": unique_admissions,
            "elevated_tests": elevated_count,
            "elevation_rate": (
                (elevated_count / len(processed["values"])) * 100
                if processed["values"]
                else 0
            ),
            "date_range": {
                "first_test": str(date_range_start) if date_range_start else None,
                "last_test": str(date_range_end) if date_range_end else None,
                "span_days": (
                    (date_range_end - date_range_start).days
                    if date_range_start and date_range_end
                    else 0
                ),
            },
            "temporal_pattern": temporal_pattern,
            "max_value": processed["max_value"],
            "historical_pattern": self._classify_historical_pattern(
                processed["values"], patient_id
            ),
        }

        # Create summary string
        summary = f"{elevated_count}/{len(processed['values'])} elevated tests across {unique_admissions} admissions"
        if date_range_start and date_range_end:
            span_days = (date_range_end - date_range_start).days
            summary += f" over {span_days} days"

        processed["patient_analysis"] = patient_analysis
        processed["summary"] = summary

        logger.debug(
            f"[PATIENT_{patient_id}] Patient-level troponin summary: {summary}"
        )

        return processed

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
            f"[PATIENT_{patient_id}] Evaluating MI criteria using {len(values)} troponin tests from complete patient history"
        )

        # Use existing criteria check but with enhanced patient-level logging
        criteria_met, criteria_details = self._check_mi_criteria(values, patient_id)

        # Add patient-level context to criteria details
        if criteria_met:
            logger.info(f"[PATIENT_{patient_id}] *** PATIENT-LEVEL MI CRITERIA MET ***")
            logger.info(
                f"[PATIENT_{patient_id}] Criteria type: {criteria_details.get('type', 'Unknown')}"
            )
            logger.info(
                f"[PATIENT_{patient_id}] Supporting evidence: {criteria_details.get('reason', 'Not specified')}"
            )
        else:
            logger.info(f"[PATIENT_{patient_id}] Patient-level MI criteria NOT met")
            logger.debug(
                f"[PATIENT_{patient_id}] Reason: {criteria_details.get('reason', 'Insufficient evidence')}"
            )

        # Enhance criteria details with patient-level context
        criteria_details["patient_level_analysis"] = True
        criteria_details["total_tests_analyzed"] = len(values)

        return criteria_met, criteria_details

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
