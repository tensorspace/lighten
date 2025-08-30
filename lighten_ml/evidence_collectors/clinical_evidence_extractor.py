"""Clinical evidence extractor for identifying MI-related symptoms and findings."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LightenLLMClient
from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class ClinicalEvidenceExtractor(BaseEvidenceCollector):
    """Extracts clinical evidence of myocardial infarction from notes."""

    # Keywords for MI-related symptoms and findings (Criteria B.1 - Symptoms of myocardial ischemia)
    # Based on clinical guideline: chest pain, chest pressure, chest tightness, substernal discomfort,
    # burning sensation, left arm/jaw/back pain (anginal equivalents), dyspnea, diaphoresis, nausea/vomiting
    SYMPTOM_KEYWORDS = [
        # Primary chest symptoms (per guideline)
        "chest pain",
        "chest pressure",
        "chest tightness",
        "substernal discomfort",
        "substernal pain",
        "substernal pressure",
        "burning sensation",
        "chest burning",
        "chest heaviness",
        "angina",
        # Anginal equivalents (per guideline)
        "left arm pain",
        "jaw pain",
        "back pain",
        "radiat.* arm",
        "radiat.* jaw",
        "radiat.* neck",
        "radiat.* back",
        "radiat.* shoulder",
        # Associated symptoms in appropriate clinical context (per guideline)
        "dyspnea",
        "shortness of breath",
        "sob",
        "diaphoresis",
        "sweating",
        "nausea",
        "vomiting",
        "lightheaded",
        "dizzy",
        "syncope",
        "palpitations",
        "fatigue",
        "weakness",
        "indigestion",
        "heartburn",
        "epigastric pain",
        # Additional clinical descriptors
        "crushing chest pain",
        "squeezing chest pain",
        "pressure-like pain",
    ]

    # Regular expressions for symptom patterns
    SYMPTOM_PATTERNS = [
        {
            "name": "Chest Pain",
            "pattern": re.compile(
                r"chest\s+(?:pain|discomfort|pressure|tightness|heaviness|burning)",
                re.IGNORECASE,
            ),
            "mi_related": True,
        },
        {
            "name": "Substernal Pain",
            "pattern": re.compile(
                r"substernal\s+(?:pain|discomfort|pressure|tightness)", re.IGNORECASE
            ),
            "mi_related": True,
        },
        {
            "name": "Radiation to Arm",
            "pattern": re.compile(r"radiat(?:e|ing|es|ed).*?(?:arm)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Radiation to Jaw",
            "pattern": re.compile(r"radiat(?:e|ing|es|ed).*?(?:jaw)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Radiation to Neck",
            "pattern": re.compile(r"radiat(?:e|ing|es|ed).*?(?:neck)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Radiation to Back",
            "pattern": re.compile(r"radiat(?:e|ing|es|ed).*?(?:back)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Radiation to Shoulder",
            "pattern": re.compile(
                r"radiat(?:e|ing|es|ed).*?(?:shoulder)", re.IGNORECASE
            ),
            "mi_related": True,
        },
        {
            "name": "Shortness of Breath",
            "pattern": re.compile(
                r"(?:shortness of breath|sob|dyspnea)", re.IGNORECASE
            ),
            "mi_related": True,
        },
        {
            "name": "Diaphoresis",
            "pattern": re.compile(r"(?:diaphoresis|sweating)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Nausea/Vomiting",
            "pattern": re.compile(r"(?:nausea|vomiting)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Lightheaded/Dizzy",
            "pattern": re.compile(r"(?:lightheaded|dizzy|syncope)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Palpitations",
            "pattern": re.compile(
                r"(?:palpitations|heart racing|pounding)", re.IGNORECASE
            ),
            "mi_related": True,
        },
        {
            "name": "Fatigue/Weakness",
            "pattern": re.compile(r"(?:fatigue|weakness|tired)", re.IGNORECASE),
            "mi_related": True,
        },
        {
            "name": "Indigestion/Heartburn",
            "pattern": re.compile(
                r"(?:indigestion|heartburn|epigastric pain)", re.IGNORECASE
            ),
            "mi_related": True,
        },
        {
            "name": "Atypical Chest Pain",
            "pattern": re.compile(r"atypical\s+chest\s+pain", re.IGNORECASE),
            "mi_related": False,
        },
    ]

    DIAGNOSIS_PATTERNS = [
        {
            "name": "MI Diagnosis",
            "pattern": re.compile(
                r"\b(myocardial infarction|mi|n?stemi)\b", re.IGNORECASE
            ),
        }
    ]

    # Keywords for MI-related findings
    FINDING_KEYWORDS = [
        "acute coronary syndrome",
        "acs",
        "stemi",
        "nstemi",
        "myocardial infarction",
        "heart attack",
        "cardiac ischemia",
        r"ejection fraction",
        r"ef\s*[<]?\s*40%?",
        r"cardiogenic shock",
    ]

    def __init__(self):
        """Initialize the clinical evidence extractor."""
        super().__init__()

    def collect_evidence(
        self, patient_id: str, hadm_id: str, text: str
    ) -> Dict[str, Any]:
        """Collect clinical evidence from a given clinical note text."""
        log_prefix = f"[{patient_id}][{hadm_id}] [CLINICAL]"
        logger.debug(f"{log_prefix} Starting clinical evidence extraction.")

        evidence = {"symptoms": []}
        findings = []

        # Use regex patterns to find symptoms
        for definition in self.SYMPTOM_PATTERNS:
            for match in definition["pattern"].finditer(text):
                finding = {
                    "finding_type": "symptom",
                    "finding_name": definition["name"],
                    "evidence": match.group(0),
                    "mi_related": definition["mi_related"],
                    "source": "regex",
                }
                findings.append(finding)

        if findings:
            # Deduplicate findings based on name and evidence
            unique_findings = list(
                {(f["finding_name"], f["evidence"]): f for f in findings}.values()
            )
            logger.info(
                f"{log_prefix} Found {len(unique_findings)} unique clinical symptoms/findings."
            )
            logger.debug(f"{log_prefix} Details: {unique_findings}")
            evidence["symptoms"] = unique_findings
        else:
            logger.debug(f"{log_prefix} No clinical symptoms/findings found.")

        return evidence
