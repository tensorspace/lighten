"""Rule engine for Myocardial Infarction (MI) detection."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        # Initialize result components
        criteria_met = {
            "A": False,  # Biomarker criteria
            "B": False,  # Ischemia criteria
        }

        confidence = {"A": 0.0, "B": 0.0}

        evidence_items = []
        details = {"criteria_A": {}, "criteria_B": {}}

        # Evaluate Criteria A: Biomarker evidence
        a_result = self._evaluate_criteria_a(evidence.get("troponin", {}))
        criteria_met["A"] = a_result["met"]
        confidence["A"] = a_result["confidence"]
        details["criteria_A"] = a_result["details"]
        evidence_items.extend(a_result.get("evidence", []))

        # Evaluate Criteria B: Ischemia evidence
        b_result = self._evaluate_criteria_b(evidence)
        criteria_met["B"] = b_result["met"]
        confidence["B"] = b_result["confidence"]
        details["criteria_B"] = b_result["details"]
        evidence_items.extend(b_result.get("evidence", []))

        # Determine overall result
        # Special handling for single elevated troponin: requires ischemia evidence
        is_single_troponin_case = (
            details["criteria_A"].get("criteria_met") == "Single elevated troponin"
            and self.config.require_ischemia_for_single_troponin
        )

        if is_single_troponin_case:
            passed = criteria_met["A"] and criteria_met["B"]
            overall_confidence = min(confidence["A"], confidence["B"])
            details["summary"] = (
                "MI criteria met (single elevated troponin with ischemia)"
            )
        elif self.config.require_both_criteria:
            passed = criteria_met["A"] and criteria_met["B"]
            overall_confidence = min(confidence["A"], confidence["B"])
            details["summary"] = "MI criteria met (biomarker and ischemia evidence)"
        else:
            passed = criteria_met["A"] or criteria_met["B"]
            overall_confidence = max(confidence["A"], confidence["B"])
            details["summary"] = "MI criteria met (biomarker or ischemia evidence)"

        # Add summary to details
        details["summary_details"] = {
            "criteria_met": criteria_met,
            "confidence_scores": confidence,
            "overall_confidence": overall_confidence,
            "decision_threshold": self.config.confidence_thresholds["medium"],
        }

        return self._create_result(
            passed=passed,
            confidence=overall_confidence,
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

        result = {"met": False, "confidence": 0.0, "details": {}, "evidence": []}

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

        if criteria_met:
            # Assign confidence based on the type of criteria met
            if "pattern" in criteria_details.get("criteria", ""):
                confidence = 0.9
                description = f"Troponin shows {criteria_details.get('criteria')}."
            else:
                confidence = 0.75  # Lower confidence for single value
                description = "Single elevated troponin detected without clear pattern."

            result.update(
                {
                    "met": True,
                    "confidence": confidence,
                    "details": criteria_details,
                    "evidence": [
                        {
                            "type": "troponin",
                            "description": description,
                            "significance": "Meets biomarker criteria for MI.",
                            "confidence": confidence,
                            "details": criteria_details,
                        }
                    ],
                }
            )
        else:
            result["details"] = {"reason": "No troponin criteria met"}

        return result

    def _evaluate_criteria_b(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria B: Ischemia evidence.

        Args:
            evidence: Dictionary containing all evidence

        Returns:
            Dictionary with evaluation results
        """
        result = {"met": False, "confidence": 0.0, "details": {}, "evidence": []}

        # Check each type of ischemia evidence
        evidence_sources = []

        # 1. Symptoms
        if self.config.consider_clinical_symptoms:
            symptoms = evidence.get("symptoms", [])
            if symptoms:
                evidence_sources.append(
                    {
                        "type": "symptoms",
                        "count": len(symptoms),
                        "confidence": 0.7,  # Moderate confidence for symptoms alone
                        "details": symptoms,
                    }
                )

        # 2. ECG findings
        if self.config.consider_ecg_findings:
            ecg_evidence = evidence.get("ecg", {})
            ecg_findings = ecg_evidence.get("ecg_findings", [])
            mi_related_ecg = [f for f in ecg_findings if f.get("mi_related", False)]

            if mi_related_ecg:
                evidence_sources.append(
                    {
                        "type": "ecg",
                        "count": len(mi_related_ecg),
                        "confidence": 0.9,  # High confidence for ECG findings
                        "details": mi_related_ecg,
                    }
                )

        # 3. Imaging findings
        if self.config.consider_imaging_evidence:
            imaging_evidence = evidence.get("imaging", {})
            if imaging_evidence.get("wall_motion_abnormalities", False):
                evidence_sources.append(
                    {
                        "type": "imaging",
                        "confidence": 0.85,
                        "details": imaging_evidence.get("imaging_findings", []),
                    }
                )

        # 4. Angiographic findings
        if self.config.consider_angiographic_evidence:
            angio_evidence = evidence.get("angiography", {})
            if angio_evidence.get("thrombus_present", False):
                evidence_sources.append(
                    {
                        "type": "angiography",
                        "confidence": 0.95,  # Very high confidence for direct visualization
                        "details": angio_evidence.get("angiography_findings", []),
                    }
                )

        # Determine if criteria are met
        met = len(evidence_sources) >= self.config.required_ischemia_criteria

        # Calculate confidence based on evidence sources
        if evidence_sources:
            # Weighted average of confidence from all sources
            total_weight = sum(src.get("confidence", 0) for src in evidence_sources)
            avg_confidence = total_weight / len(evidence_sources)
        else:
            avg_confidence = 0.0

        # Update result
        result.update(
            {
                "met": met,
                "confidence": avg_confidence,
                "details": {
                    "evidence_sources": [
                        {
                            "type": src["type"],
                            "confidence": src["confidence"],
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
                        "confidence": avg_confidence,
                        "details": {
                            "sources": [src["type"] for src in evidence_sources],
                            "required": self.config.required_ischemia_criteria,
                        },
                    }
                ],
            }
        )

        return result
