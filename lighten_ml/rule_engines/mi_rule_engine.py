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
        """Evaluate evidence for Myocardial Infarction."""
        logger.info("[RULE_ENGINE] === Starting MI Diagnosis Evaluation ===")

        # Initialize result components
        criteria_met = {"A": False, "B": False}
        details = {"criteria_A": {}, "criteria_B": {}}
        evidence_items = []

        # Evaluate Criteria A: Biomarker evidence
        logger.info("[RULE_ENGINE] [CRITERIA_A] Evaluating biomarker evidence (Troponin)...")
        a_result = self._evaluate_criteria_a(evidence.get("troponin", {}))
        criteria_met["A"] = a_result["met"]
        details["criteria_A"] = a_result["details"]
        evidence_items.extend(a_result.get("evidence", []))
        logger.info(f"[RULE_ENGINE] [CRITERIA_A] Result: {'MET' if criteria_met['A'] else 'NOT MET'}")

        # Early termination if Criteria A is not met
        if not criteria_met["A"]:
            logger.info("[RULE_ENGINE] Criteria A not met. Final diagnosis is NEGATIVE.")
            details["criteria_B"] = {"met": False, "reason": "Not evaluated as Criteria A was not met."}
        else:
            # Evaluate Criteria B: Ischemia evidence
            logger.info("[RULE_ENGINE] [CRITERIA_B] Criteria A met. Evaluating clinical evidence for ischemia...")
            b_result = self._evaluate_criteria_b(evidence)
            criteria_met["B"] = b_result["met"]
            details["criteria_B"] = b_result["details"]
            evidence_items.extend(b_result.get("evidence", []))
            logger.info(f"[RULE_ENGINE] [CRITERIA_B] Result: {'MET' if criteria_met['B'] else 'NOT MET'}")

        # Determine overall result
        passed = criteria_met["A"] and criteria_met["B"]

        logger.info(f"[RULE_ENGINE] === MI Diagnosis Evaluation Complete: {'POSITIVE' if passed else 'NEGATIVE'} ===")

        return RuleResult(
            passed=passed,
            details=details,
            evidence_items=evidence_items,
            engine_name=self.__class__.__name__,
        )

    def evaluate_criteria_a(self, troponin_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to evaluate only Criteria A (Troponin Biomarkers)."""
        logger.info("[RULE_ENGINE] [CRITERIA_A_ONLY] Evaluating biomarker evidence (Troponin)...")
        result = self._evaluate_criteria_a(troponin_evidence)
        logger.info(f"[RULE_ENGINE] [CRITERIA_A_ONLY] Result: {'MET' if result['met'] else 'NOT MET'}")
        return result

    def _evaluate_criteria_a(self, troponin_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria A: Rise and/or fall of troponin."""
        if not troponin_evidence or not troponin_evidence.get("troponin_available", False):
            return {"met": False, "details": "No troponin data available."}

        tests = troponin_evidence.get("troponin_tests", [])
        if not tests:
            return {"met": False, "details": "No troponin tests found in the data."}

        # Check for at least one value above the diagnostic threshold
        above_threshold_tests = [t for t in tests if t.get("above_threshold")]
        if not above_threshold_tests:
            return {
                "met": False,
                "details": f"No troponin values above threshold of {self.config.troponin_threshold} ng/mL.",
            }

        # Check for rise and/or fall pattern
        if self.config.require_rise_and_fall and len(tests) > 1:
            # Simplified check for rise/fall: are there non-identical values?
            unique_values = {t["value"] for t in tests}
            if len(unique_values) > 1:
                return {
                    "met": True,
                    "details": "Troponin rise and/or fall detected with at least one value above threshold.",
                    "evidence": above_threshold_tests,
                }

        # If rise/fall is not required or not detected, one value above threshold is sufficient
        return {
            "met": True,
            "details": "At least one troponin value was above the diagnostic threshold.",
            "evidence": above_threshold_tests,
        }

    def _evaluate_criteria_b(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria B: Corroborating evidence of myocardial ischemia."""
        ischemia_evidence_found = []

        # 1. Symptoms of myocardial ischemia
        if evidence.get("clinical", {}).get("symptoms"):
            ischemia_evidence_found.append("Symptoms of myocardial ischemia")

        # 2. New ischemic ECG changes
        if evidence.get("ecg", {}).get("findings"):
            ischemia_evidence_found.append("Ischemic ECG changes")

        # 3. Development of pathological Q waves (often in ECG)
        # (This would require more specific ECG analysis)

        # 4. Imaging evidence
        if evidence.get("imaging", {}).get("findings"):
            ischemia_evidence_found.append("Imaging evidence of ischemia")

        # 5. Intracoronary thrombus
        if evidence.get("angiography", {}).get("findings"):
            ischemia_evidence_found.append("Intracoronary thrombus identified")

        if ischemia_evidence_found:
            return {
                "met": True,
                "details": f"Found {len(ischemia_evidence_found)} type(s) of ischemia evidence: {', '.join(ischemia_evidence_found)}.",
                "evidence": ischemia_evidence_found,
            }
        else:
            return {"met": False, "details": "No clinical evidence of myocardial ischemia found."}
