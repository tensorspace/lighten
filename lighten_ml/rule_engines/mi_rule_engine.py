"""Rule engine for Myocardial Infarction (MI) detection."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base_rule_engine import BaseRuleEngine, RuleResult

logger = logging.getLogger(__name__)


@dataclass
class MIRuleEngineConfig:
    """Configuration for the MI Rule Engine."""

    # Troponin threshold in ng/mL (Troponin T)
    troponin_threshold: float = 0.014

    # Threshold for considering a single high troponin value (5x upper limit of normal)
    single_value_threshold: float = 0.07  # 5 * 0.014

    # Minimum percentage increase for significant rise (50%)
    min_percent_increase: float = 50.0

    # Minimum percentage decrease for significant fall (25%)
    min_percent_decrease: float = 25.0

    # Minimum number of troponin measurements required for trend analysis
    min_troponin_measurements: int = 2

    # Time window in hours to look for dynamic changes
    dynamic_change_window_hours: int = 72

    # Required number of criteria from group B (ischemia evidence)
    required_ischemia_criteria: int = 1

    # Whether to require both criteria A and B for MI diagnosis
    require_both_criteria: bool = True

    # Whether to consider clinical symptoms as evidence
    consider_clinical_symptoms: bool = True

    # Whether to consider ECG findings as evidence
    consider_ecg_findings: bool = True

    # Whether to consider imaging findings as evidence
    consider_imaging_findings: bool = True

    # Whether to consider angiographic findings as evidence
    consider_angiographic_findings: bool = True

    # Confidence thresholds
    confidence_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"high": 0.9, "medium": 0.7, "low": 0.5}
    )

    # Default confidence level when evidence is missing
    default_confidence: float = 0.5

    # Require ischemia evidence for single high troponin
    require_ischemia_for_single_troponin: bool = True

    # Whether to consider imaging findings for Criteria B
    consider_imaging_evidence: bool = True

    # Whether to consider angiographic findings for Criteria B
    consider_angiographic_evidence: bool = True


class MIRuleEngine(BaseRuleEngine[MIRuleEngineConfig]):
    """Rule engine for detecting Myocardial Infarction based on clinical evidence.

    Implements the 4th Universal Definition of Myocardial Infarction requiring:
    - Criteria A: Detection of rise/fall of cardiac troponin with at least one value above diagnostic threshold
    - Criteria B: At least ONE of 5 evidence types of myocardial ischemia:
        1. Symptoms of myocardial ischemia
        2. New ischemic ECG changes (ST elevation/depression, T wave inversion)
        3. Development of pathological Q waves
        4. Imaging evidence of new loss of viable myocardium or wall motion abnormality
        5. Identification of intracoronary thrombus by angiography or autopsy

    Both Criteria A AND B must be met for MI diagnosis.
    """

    def __init__(self, config: Optional[MIRuleEngineConfig] = None):
        """Initialize the MI Rule Engine.

        Args:
            config: Configuration for the rule engine. If None, defaults will be used.
        """
        super().__init__(config or MIRuleEngineConfig())

    def evaluate(self, evidence: Dict[str, Any]) -> RuleResult:
        """Evaluate evidence for Myocardial Infarction.

        Args:
            evidence: Dictionary containing evidence from data sources

        Returns:
            RuleResult with the evaluation
        """
        logger.info("[MI_EVALUATION] Starting MI diagnosis evaluation")
        logger.debug(f"[DEBUG] Evidence categories available: {list(evidence.keys())}")

        # Log evidence summary for debugging
        troponin_data = evidence.get("troponin", {})
        clinical_data = evidence.get("clinical", {})
        logger.debug(
            f"[DEBUG] Troponin available: {troponin_data.get('troponin_available', False)}"
        )
        logger.debug(
            f"[DEBUG] Troponin tests count: {len(troponin_data.get('troponin_tests', []))}"
        )
        logger.debug(
            f"[DEBUG] Clinical symptoms count: {len(clinical_data.get('symptoms', []))}"
        )
        logger.debug(
            f"[DEBUG] Clinical diagnoses count: {len(clinical_data.get('diagnoses', []))}"
        )

        # Initialize result components
        criteria_met = {
            "A": False,  # Biomarker criteria
            "B": False,  # Ischemia criteria
        }

        evidence_items = []
        details = {"criteria_A": {}, "criteria_B": {}}

        # Evaluate Criteria A: Biomarker evidence
        logger.info("[CRITERIA_A] Starting biomarker evidence evaluation...")
        logger.debug(f"[DEBUG] Troponin threshold: {self.config.troponin_threshold}")
        logger.debug(
            f"[DEBUG] Single value threshold: {self.config.single_value_threshold}"
        )

        a_result = self._evaluate_criteria_a(evidence.get("troponin", {}))
        criteria_met["A"] = a_result["met"]
        details["criteria_A"] = a_result["details"]
        evidence_items.extend(a_result.get("evidence", []))

        logger.info(f"[CRITERIA_A] Result: {'MET' if criteria_met['A'] else 'NOT MET'}")
        logger.debug(f"[DEBUG] Criteria A details: {a_result['details']}")

        # Early termination optimization: Skip Criteria B if A is not met
        # Clinical guideline requires BOTH A AND B, so no point evaluating B if A fails
        if not criteria_met["A"]:
            logger.info(
                "[EARLY_TERMINATION] Criteria A not met - skipping Criteria B evaluation"
            )
            logger.info(
                "[PERFORMANCE] Computational resources saved by early termination"
            )
            logger.info(
                "[CLINICAL_GUIDELINE] Both A AND B required - early exit when A fails"
            )

            # Set default values for criteria B (not evaluated)
            criteria_met["B"] = False
            details["criteria_B"] = {
                "reason": "Not evaluated - Criteria A failed",
                "early_termination": True,
                "performance_optimization": "Skipped expensive ischemia analysis",
            }
        else:
            # Evaluate Criteria B: Ischemia evidence (only if A is met)
            logger.info(
                "[CRITERIA_B] Criteria A met - proceeding to evaluate ischemia evidence..."
            )
            b_result = self._evaluate_criteria_b(evidence)
            criteria_met["B"] = b_result["met"]
            details["criteria_B"] = b_result["details"]
            evidence_items.extend(b_result.get("evidence", []))

            logger.info(
                f"[CRITERIA_B] Result: {'MET' if criteria_met['B'] else 'NOT MET'}"
            )

        # Determine overall result
        # Special handling for single elevated troponin: requires ischemia evidence
        is_single_troponin_case = (
            details["criteria_A"].get("criteria_met") == "Single elevated troponin"
            and self.config.require_ischemia_for_single_troponin
        )

        logger.info("[FINAL_DECISION] === FINAL MI RULE ENGINE DECISION ===")
        logger.info("[FINAL_DECISION] 🎯 Cross-admission evidence evaluation complete")
        logger.info(
            f"[FINAL_DECISION] 📊 Criteria A (Troponin): {'✅ MET' if criteria_met['A'] else '❌ NOT MET'}"
        )
        logger.info(
            f"[FINAL_DECISION] 🩺 Criteria B (Clinical): {'✅ MET' if criteria_met['B'] else '❌ NOT MET'}"
        )
        logger.info(
            f"[FINAL_DECISION] 🔍 Single troponin case: {'Yes' if is_single_troponin_case else 'No'}"
        )
        logger.info(
            f"[FINAL_DECISION] 📋 Guideline requirement: {'Both A AND B required' if self.config.require_both_criteria else 'Alternative logic'}"
        )
        logger.info(
            f"[FINAL_DECISION] 🏥 Evidence source: Cross-admission patient timeline analysis"
        )

        if is_single_troponin_case:
            passed = criteria_met["A"] and criteria_met["B"]
            details["summary"] = (
                "MI criteria met (single elevated troponin with ischemia)"
            )
            logger.info(
                f"[SINGLE_TROPONIN] 🎯 Single elevated troponin case - ischemia evidence required"
            )
            logger.info(
                f"[SINGLE_TROPONIN] 📋 4th Universal Definition compliance: Single troponin + clinical evidence"
            )
            logger.info(
                f"[SINGLE_TROPONIN] 🔬 Final evaluation: A={criteria_met['A']}, B={criteria_met['B']}, Result={'✅ POSITIVE' if passed else '❌ NEGATIVE'}"
            )
        elif self.config.require_both_criteria:
            passed = criteria_met["A"] and criteria_met["B"]
            details["summary"] = "MI criteria met (biomarker and ischemia evidence)"
            logger.info(
                f"[STANDARD_CASE] 📈 Standard MI criteria: troponin rise/fall pattern detected"
            )
            logger.info(
                f"[STANDARD_CASE] 🎯 4th Universal Definition compliance: Dynamic troponin changes"
            )
            logger.info(
                f"[STANDARD_CASE] 🔬 Final evaluation: A={criteria_met['A']}, B={criteria_met['B']}, Result={'✅ POSITIVE' if passed else '❌ NEGATIVE'}"
            )
        else:
            passed = criteria_met["A"] or criteria_met["B"]
            details["summary"] = "MI criteria met (biomarker or ischemia evidence)"
            logger.info(
                f"FINAL DEBUG - Either criteria: A={criteria_met['A']}, B={criteria_met['B']}, Result={passed}"
            )

        logger.info(
            f"[FINAL_DECISION] *** 🚨 OVERALL MI DIAGNOSIS: {'✅ POSITIVE' if passed else '❌ NEGATIVE'} ***"
        )
        logger.info(
            f"[FINAL_DECISION] 📊 Cross-admission analysis {'SUPPORTS' if passed else 'DOES NOT SUPPORT'} MI diagnosis"
        )
        if passed:
            logger.info(
                f"[FINAL_DECISION] 🎉 MI criteria satisfied using patient-level evidence aggregation"
            )
        else:
            logger.info(
                f"[FINAL_DECISION] ⚠️ MI criteria not met despite comprehensive cross-admission analysis"
            )

        # Add summary to details
        details["summary_details"] = {
            "criteria_met": criteria_met,
        }

        return self._create_result(
            passed=passed,
            confidence=0.0,  # Confidence removed - no longer calculated
            evidence=evidence_items,
            details=details,
        )

    def _evaluate_criteria_a(self, troponin_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria A: Biomarker evidence.

        Args:
            troponin_evidence: Dictionary containing troponin test results

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=== RULE ENGINE CRITERIA A (BIOMARKER) EVALUATION ===")

        result = {"met": False, "details": {}, "evidence": []}

        logger.info(
            f"Troponin evidence available: {troponin_evidence.get('troponin_available', False)}"
        )
        logger.info(f"Troponin evidence keys: {list(troponin_evidence.keys())}")

        if not troponin_evidence or not troponin_evidence.get(
            "troponin_available", False
        ):
            logger.warning("CRITERIA A FAILED: No troponin data available")
            result["details"] = {"reason": "No troponin data available"}
            result["evidence"].append(
                {
                    "type": "troponin",
                    "description": "Troponin test results",
                    "significance": "No troponin data available",
                    "confidence": 0.0,
                }
            )
            return result

        # Use the pre-calculated result from TroponinAnalyzer
        criteria_met = troponin_evidence.get("mi_criteria_met", False)
        criteria_details = troponin_evidence.get("criteria_details", {})

        # DETAILED LOGGING FOR DEBUGGING
        logger.info(f"CRITERIA A DEBUG - mi_criteria_met: {criteria_met}")
        logger.info(f"CRITERIA A DEBUG - criteria_details: {criteria_details}")
        logger.info(
            f"CRITERIA A DEBUG - max_troponin: {troponin_evidence.get('max_troponin', 'N/A')}"
        )
        logger.info(
            f"CRITERIA A DEBUG - troponin_tests count: {len(troponin_evidence.get('troponin_tests', []))}"
        )

        # Log individual troponin values for debugging with unit conversion details
        troponin_tests = troponin_evidence.get("troponin_tests", [])
        if troponin_tests:
            logger.info(
                f"CRITERIA A DEBUG - Individual troponin values with unit conversion:"
            )
            for i, test in enumerate(troponin_tests[:5]):  # Log first 5 tests
                original_value = test.get("original_value", "N/A")
                original_unit = test.get("original_unit", "N/A")
                converted_value = test.get("value", "N/A")
                converted_unit = test.get("unit", "N/A")
                above_threshold = test.get("above_threshold", "N/A")
                threshold_analysis = test.get("threshold_analysis", {})
                fold_change = threshold_analysis.get("fold_change", "N/A")

                logger.info(f"  [TEST] Test {i+1}:")
                logger.info(f"    [DATA] Original: {original_value} {original_unit}")
                logger.info(f"    [DATA] Converted: {converted_value} {converted_unit}")
                logger.info(f"    [DATA] Above threshold: {above_threshold}")

                # Handle NaN fold_change values
                if isinstance(fold_change, (int, float)) and not (
                    fold_change != fold_change
                ):  # Check for NaN
                    logger.info(f"    [DATA] Fold change: {fold_change}")
                else:
                    logger.info(f"    [DATA] Fold change: N/A (invalid calculation)")

                logger.info(f"    [DATA] Timestamp: {test.get('timestamp', 'N/A')}")
        else:
            logger.warning("CRITERIA A DEBUG - No troponin_tests found in evidence")

        if criteria_met:
            # Determine description based on the type of criteria met
            if "pattern" in criteria_details.get("criteria", ""):
                description = f"Troponin shows {criteria_details.get('criteria')}."
            else:
                description = "Single elevated troponin detected without clear pattern."

            result.update(
                {
                    "met": True,
                    "details": criteria_details,
                    "evidence": [
                        {
                            "type": "troponin",
                            "description": description,
                            "significance": "Meets biomarker criteria for MI.",
                            "details": criteria_details,
                        }
                    ],
                }
            )
        else:
            logger.warning("CRITERIA A FAILED: Troponin criteria not met")
            logger.warning(f"CRITERIA A FAILED - mi_criteria_met was: {criteria_met}")
            logger.warning(
                f"CRITERIA A FAILED - criteria_details was: {criteria_details}"
            )
            result["details"] = {
                "reason": "No troponin criteria met",
                "debug_info": criteria_details,
            }

        logger.info(f"CRITERIA A FINAL RESULT: met={result['met']}")
        return result

    def _evaluate_criteria_b(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria B: Ischemia evidence.

        Args:
            evidence: Dictionary containing all evidence

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=== RULE ENGINE CRITERIA B (ISCHEMIA) EVALUATION ===")

        result = {"met": False, "details": {}, "evidence": []}

        # Log available evidence types for debugging
        logger.info(
            f"CRITERIA B DEBUG - Available evidence keys: {list(evidence.keys())}"
        )
        logger.info(
            f"CRITERIA B DEBUG - Clinical symptoms available: {'symptoms' in evidence}"
        )
        logger.info(f"CRITERIA B DEBUG - ECG evidence available: {'ecg' in evidence}")
        logger.info(
            f"CRITERIA B DEBUG - Imaging evidence available: {'imaging' in evidence}"
        )
        logger.info(
            f"CRITERIA B DEBUG - Angiography evidence available: {'angiography' in evidence}"
        )

        # Check each type of ischemia evidence
        evidence_sources = []

        # 1. Symptoms
        if self.config.consider_clinical_symptoms:
            # Fix: symptoms are nested under 'clinical' key
            clinical_evidence = evidence.get("clinical", {})
            symptoms = clinical_evidence.get("symptoms", [])
            logger.info(f"CRITERIA B DEBUG - Symptoms found: {len(symptoms)}")
            if symptoms:
                logger.info(
                    f"CRITERIA B DEBUG - First few symptoms: {[s.get('symptom', s.get('name', 'unknown')) for s in symptoms[:3]]}"
                )
                evidence_sources.append(
                    {
                        "type": "symptoms",
                        "count": len(symptoms),
                        "details": symptoms,
                    }
                )
            else:
                logger.info("CRITERIA B DEBUG - No symptoms found in evidence")

        # 2. ECG findings
        if self.config.consider_ecg_findings:
            ecg_evidence = evidence.get("ecg", {})
            ecg_findings = ecg_evidence.get("ecg_findings", [])
            mi_related_ecg = [f for f in ecg_findings if f.get("mi_related", False)]

            logger.info(f"CRITERIA B DEBUG - ECG findings total: {len(ecg_findings)}")
            logger.info(
                f"CRITERIA B DEBUG - MI-related ECG findings: {len(mi_related_ecg)}"
            )
            if mi_related_ecg:
                logger.info(
                    f"CRITERIA B DEBUG - ECG findings: {[f.get('finding', 'unknown') for f in mi_related_ecg[:3]]}"
                )
                evidence_sources.append(
                    {
                        "type": "ecg",
                        "count": len(mi_related_ecg),
                        "details": mi_related_ecg,
                    }
                )
            else:
                logger.info("CRITERIA B DEBUG - No MI-related ECG findings found")

        # 3. Imaging findings
        if self.config.consider_imaging_evidence:
            imaging_evidence = evidence.get("imaging", {})
            wall_motion_abnormalities = imaging_evidence.get(
                "wall_motion_abnormalities", False
            )
            imaging_findings = imaging_evidence.get("imaging_findings", [])

            logger.info(
                f"CRITERIA B DEBUG - Wall motion abnormalities: {wall_motion_abnormalities}"
            )
            logger.info(
                f"CRITERIA B DEBUG - Imaging findings count: {len(imaging_findings)}"
            )

            if wall_motion_abnormalities:
                logger.info(
                    "CRITERIA B DEBUG - Wall motion abnormalities found - adding to evidence"
                )
                evidence_sources.append(
                    {
                        "type": "imaging",
                        "details": imaging_findings,
                    }
                )
            else:
                logger.info("CRITERIA B DEBUG - No wall motion abnormalities found")

        # 4. Angiographic findings
        if self.config.consider_angiographic_evidence:
            angio_evidence = evidence.get("angiography", {})
            thrombus_present = angio_evidence.get("thrombus_present", False)
            angio_findings = angio_evidence.get("angiography_findings", [])

            logger.info(f"CRITERIA B DEBUG - Thrombus present: {thrombus_present}")
            logger.info(
                f"CRITERIA B DEBUG - Angiography findings count: {len(angio_findings)}"
            )

            if thrombus_present:
                logger.info("CRITERIA B DEBUG - Thrombus found - adding to evidence")
                evidence_sources.append(
                    {
                        "type": "angiography",
                        "details": angio_findings,
                    }
                )
            else:
                logger.info("CRITERIA B DEBUG - No thrombus found")

        # Determine if criteria are met
        met = len(evidence_sources) >= self.config.required_ischemia_criteria

        # FINAL CRITERIA B LOGGING
        logger.info(
            f"CRITERIA B DEBUG - Total evidence sources found: {len(evidence_sources)}"
        )
        logger.info(
            f"CRITERIA B DEBUG - Required ischemia criteria: {self.config.required_ischemia_criteria}"
        )
        logger.info(
            f"CRITERIA B DEBUG - Evidence source types: {[src['type'] for src in evidence_sources]}"
        )
        logger.info(f"CRITERIA B DEBUG - Criteria B met: {met}")

        # Update result
        result.update(
            {
                "met": met,
                "details": {
                    "evidence_sources": [
                        {
                            "type": src["type"],
                            "count": src.get("count", 1),
                        }
                        for src in evidence_sources
                    ],
                    "required_sources": self.config.required_ischemia_criteria,
                    "found_sources": len(evidence_sources),
                },
                "evidence": [
                    {
                        "type": "ischemia_evidence",
                        "description": f"Found {len(evidence_sources)} source(s) of ischemia evidence",
                        "significance": (
                            "Meets ischemia criteria"
                            if met
                            else "Insufficient ischemia evidence"
                        ),
                        "details": {
                            "sources": [src["type"] for src in evidence_sources],
                            "required": self.config.required_ischemia_criteria,
                        },
                    }
                ],
            }
        )

        logger.info(f"CRITERIA B FINAL RESULT: met={result['met']}")
        return result
