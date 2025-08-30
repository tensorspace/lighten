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

    def __init__(
        self,
        notes_data_loader: Any,
        llm_client: Optional[Any] = None,
        max_notes: Optional[int] = None,
    ):
        """Initialize the clinical evidence extractor.

        Args:
            notes_data_loader: Instance of ClinicalNotesDataLoader
            llm_client: Optional instance of the LLM client
            max_notes: Maximum number of notes to process
        """
        super().__init__(
            notes_data_loader=notes_data_loader,
            llm_client=llm_client,
            max_notes=max_notes,
        )

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect clinical evidence from notes for a specific admission."""
        log_prefix = f"[{patient_id}][{hadm_id}] [CLINICAL_EXTRACTOR]"

        evidence = self._get_evidence_base()
        evidence["symptoms"] = []
        evidence["metadata"] = {"extraction_mode": "none"}

        logger.info(f"{log_prefix} Fetching clinical notes for symptom evidence.")
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)

        if notes.empty:
            logger.warning(f"{log_prefix} No clinical notes found for patient.")
            return evidence

        # Limit number of notes if configured
        if self.max_notes and len(notes) > self.max_notes:
            logger.info(f"{log_prefix} Limiting notes from {len(notes)} to {self.max_notes}.")
            notes = notes.head(self.max_notes)

        logger.info(f"{log_prefix} Processing {len(notes)} clinical notes.")

        try_llm = self.llm_client and self.llm_client.enabled
        logger.info(f"{log_prefix} Method selection: LLM available={try_llm}")

        extracted_with_llm = False
        if try_llm:
            try:
                logger.info(f"{log_prefix} Attempting LLM extraction...")
                findings = self._extract_with_llm(notes)
                evidence["symptoms"] = findings
                evidence["metadata"]["extraction_mode"] = "llm"
                extracted_with_llm = True
                logger.info(f"{log_prefix} LLM extraction successful: {len(findings)} findings")
            except Exception as e:
                logger.warning(f"{log_prefix} LLM extraction failed: {e}")

        if not extracted_with_llm:
            logger.info(f"{log_prefix} Falling back to regex extraction...")
            findings = self._extract_with_regex(notes)
            evidence["symptoms"] = findings
            evidence["metadata"]["extraction_mode"] = "regex"
            logger.info(f"{log_prefix} Regex extraction completed: {len(findings)} findings")

        return evidence
