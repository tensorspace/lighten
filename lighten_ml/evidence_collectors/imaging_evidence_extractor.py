"""Evidence extractor for imaging findings related to MI."""

import logging
import re
from typing import Any, Dict, List

from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class ImagingEvidenceExtractor(BaseEvidenceCollector):
    """Extracts imaging evidence (e.g., from echocardiograms) for MI."""

    IMAGING_PATTERNS = [
        {
            "name": "Wall Motion Abnormality",
            "pattern": re.compile(
                r"(wall motion abnormalit(y|ies)|WMA|hypokinesis|akinesis|dyskinesis)",
                re.IGNORECASE,
            ),
            "mi_related": True,
        },
        {
            "name": "Loss of Viable Myocardium",
            "pattern": re.compile(
                r"(loss of.*viab|non-viable|infarct|scar|late gadolinium enhancement|lge)",
                re.IGNORECASE,
            ),
            "mi_related": True,
        },
    ]

    def __init__(self):
        """Initialize the imaging evidence extractor."""
        super().__init__()

    def collect_evidence(
        self, patient_id: str, hadm_id: str, text: str
    ) -> Dict[str, Any]:
        """Collect imaging evidence from a given clinical note text."""
        log_prefix = f"[{patient_id}][{hadm_id}] [IMAGING]"
        logger.debug(f"{log_prefix} Starting imaging evidence extraction.")

        evidence = {"imaging_findings": []}
        findings = []

        for definition in self.IMAGING_PATTERNS:
            for match in definition["pattern"].finditer(text):
                finding = {
                    "finding_type": "imaging",
                    "finding_name": definition["name"],
                    "evidence": match.group(0),
                    "mi_related": definition["mi_related"],
                    "source": "regex",
                }
                findings.append(finding)

        if findings:
            logger.info(f"{log_prefix} Found {len(findings)} imaging-related findings.")
            logger.debug(f"{log_prefix} Details: {findings}")

        evidence["imaging_findings"] = findings
        return evidence
