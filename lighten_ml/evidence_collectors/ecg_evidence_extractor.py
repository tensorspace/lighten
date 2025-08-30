"""ECG evidence extractor for identifying MI-related ECG findings."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class ECGEvidenceExtractor(BaseEvidenceCollector):
    """Extracts ECG evidence related to myocardial infarction."""

    # ECG patterns indicative of MI
    ECG_PATTERNS = [
        # ST elevation (Criteria B.2 - New ischemic ECG changes)
        {
            "name": "ST Elevation",
            "pattern": re.compile(
                r"(?:new\s+)?ST\s*(?:segment\s*)?elevat|STEMI", re.IGNORECASE
            ),
            "criteria": "New ST elevation ≥1mm in two contiguous leads documented/interpreted by provider",
            "mi_related": True,
        },
        # ST depression (Criteria B.2 - New ischemic ECG changes)
        {
            "name": "ST Depression",
            "pattern": re.compile(
                r"(?:new\s+)?ST\s*(?:segment\s*)?depress|(?:horizontal|down-sloping)\s+ST",
                re.IGNORECASE,
            ),
            "criteria": "New horizontal or down-sloping ST depression ≥0.5mm in two contiguous leads documented/interpreted by provider",
            "mi_related": True,
        },
        # T wave inversion (Criteria B.2 - New ischemic ECG changes)
        {
            "name": "T Wave Inversion",
            "pattern": re.compile(
                r"(?:new\s+)?T\s*wave\s*inver|inverted\s+T\s+waves?", re.IGNORECASE
            ),
            "criteria": "New T wave inversion ≥1mm in two contiguous leads with prominent R wave or R/S ratio >1 documented/interpreted by provider",
            "mi_related": True,
        },
        # Pathologic Q waves (Criteria B.3 - Development of pathological Q waves)
        {
            "name": "Pathologic Q Waves",
            "pattern": re.compile(
                r"(?:new|pathologic|abnormal|significant)\s*q\s*waves?|qs\s*complex",
                re.IGNORECASE,
            ),
            "criteria": "New Q waves ≥0.02 seconds or QS complex in ≥2 contiguous leads documented by provider",
            "mi_related": True,
        },
        # Other ECG findings
        {
            "name": "Left Bundle Branch Block",
            "pattern": re.compile(
                r"left\s+bundle\s+branch\s+block|LBBB", re.IGNORECASE
            ),
            "criteria": "New or presumably new LBBB",
            "mi_related": True,
        },
        {
            "name": "Ventricular Arrhythmia",
            "pattern": re.compile(
                r"ventricular\s+(?:tachycardia|fibrillation)|VT\/VF", re.IGNORECASE
            ),
            "criteria": "Ventricular tachycardia or fibrillation",
            "mi_related": True,
        },
        {
            "name": "Atrial Fibrillation",
            "pattern": re.compile(r"atrial\s+fibrillation|a\s*?fib", re.IGNORECASE),
            "criteria": "New onset atrial fibrillation",
            "mi_related": False,
        },
    ]

    # Lead groups for contiguous leads
    LEAD_GROUPS = {
        "inferior": ["II", "III", "aVF"],
        "lateral": ["I", "aVL", "V5", "V6"],
        "anterior": ["V1", "V2", "V3", "V4"],
        "septal": ["V1", "V2"],
        "anteroseptal": ["V1", "V2", "V3", "V4"],
        "anterolateral": ["I", "aVL", "V3", "V4", "V5", "V6"],
        "inferolateral": ["II", "III", "aVF", "V5", "V6"],
        "inferoposterior": ["II", "III", "aVF", "V7", "V8", "V9"],
        "posterior": ["V7", "V8", "V9"],
    }

    # Thresholds for validation
    VALIDATION_THRESHOLDS = {
        "ST Elevation": {"mm": 1.0},
        "ST Depression": {"mm": 0.5},
        "T Wave Inversion": {"mm": 1.0},
    }

    def __init__(
        self,
        notes_data_loader: Any,
        llm_client: Optional[Any] = None,
        max_notes: Optional[int] = None,
    ):
        """Initialize the ECG evidence extractor.

        Args:
            notes_data_loader: Instance of ClinicalNotesDataLoader for accessing clinical notes
            llm_client: Optional instance of the LLM client
            max_notes: Maximum number of notes to process
        """
        super().__init__(
            notes_data_loader=notes_data_loader,
            llm_client=llm_client,
            max_notes=max_notes,
        )

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect ECG evidence from notes for a specific admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing ECG evidence
        """
        evidence = self._get_evidence_base()
        evidence["ecg_findings"] = []
        evidence["metadata"] = {"extraction_mode": "none"}

        # Get clinical notes for the admission
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)

        if notes.empty:
            return evidence

        # Filter for ECG notes
        ecg_notes = notes[notes["note_type"] == "ECG"]

        # Limit number of notes if configured
        if self.max_notes and len(ecg_notes) > self.max_notes:
            ecg_notes = ecg_notes.head(self.max_notes)

        if ecg_notes.empty:
            evidence["metadata"]["extraction_mode"] = "none_no_notes"
            return evidence

        # LLM-first with regex fallback
        logger.info(
            f"[{hadm_id}] ECG EXTRACTION - Method selection: LLM available={self.llm_client and self.llm_client.enabled}"
        )

        extracted_with_llm = False
        if self.llm_client and self.llm_client.enabled:
            try:
                logger.info(
                    f"[{hadm_id}] ECG EXTRACTION - Attempting LLM extraction..."
                )
                llm_findings = self._extract_with_llm(ecg_notes)
                # Post-process and validate LLM findings
                ecg_findings = self._post_process_llm_findings(llm_findings)
                evidence["metadata"]["extraction_mode"] = "llm_validated"
                extracted_with_llm = True
                logger.info(
                    f"[{hadm_id}] ECG EXTRACTION - LLM extraction successful: {len(ecg_findings)} findings"
                )
            except Exception as e:
                # Fallback to regex if LLM fails
                logger.warning(
                    f"[{hadm_id}] ECG EXTRACTION - LLM extraction failed: {str(e)}, falling back to regex"
                )

        # Fallback to regex if LLM is not used or fails
        if not extracted_with_llm:
            evidence["metadata"]["extraction_mode"] = (
                "regex_fallback" if self.llm_client else "regex"
            )
            logger.info(f"[{hadm_id}] ECG EXTRACTION - Using regex extraction")
            ecg_findings = self._extract_findings_regex(ecg_notes)
            logger.info(
                f"[{hadm_id}] ECG EXTRACTION - Regex extraction complete: {len(ecg_findings)} findings"
            )

        # Ensure every ECG finding dictionary has the 'mi_related' key
        for finding in ecg_findings:
            finding["mi_related"] = finding.get("mi_related", False)

        # Log detailed ECG evidence found
        mi_related_findings = [f for f in ecg_findings if f.get("mi_related", False)]
        if mi_related_findings:
            logger.info(f"[{hadm_id}] ECG EVIDENCE FOUND (MI-related):")
            for i, finding in enumerate(
                mi_related_findings[:5], 1
            ):  # Log first 5 MI-related findings
                logger.info(
                    f"[{hadm_id}]   {i}. {finding.get('finding', 'unknown')} in {finding.get('leads', 'unknown leads')}"
                )
                logger.info(
                    f"[{hadm_id}]      Confidence: {finding.get('confidence', 'N/A')}"
                )
                logger.info(
                    f"[{hadm_id}]      Context: {finding.get('context', 'N/A')[:100]}..."
                )
        else:
            logger.info(f"[{hadm_id}] ECG EVIDENCE - No MI-related findings detected")

        evidence.update(
            {
                "ecg_findings": ecg_findings,
                "sources": [
                    {"type": "clinical_notes", "ecg_finding_count": len(ecg_findings)}
                ],
            }
        )

        return evidence

    def _extract_with_llm(self, notes: Any) -> List[Dict[str, Any]]:
        """Use LLM to extract ECG findings structured for MI criteria."""
        instructions = (
            "Read the ECG report and extract findings relevant to myocardial infarction. "
            "Return a JSON object with a single key 'ecg_findings', which is a list of objects. "
            "Each object must have the following keys: 'finding' (e.g., 'ST Elevation', 'T Wave Inversion'), "
            "'context' (the sentence where the finding was mentioned), 'leads' (a list of lead names like 'V1', 'aVL'), "
            "'measurements' (an object with values like {'st_elevation_mm': 1.5}), and 'is_new' (a boolean). "
            "Focus only on MI-related changes like ST elevation/depression, T wave inversions, and Q waves."
        )
        findings: List[Dict[str, Any]] = []
        for _, note in notes.iterrows():
            text = str(note.get("text", ""))
            if not text.strip():
                continue
            data = self.llm_client.extract_json(instructions, text)
            for f in data.get("ecg_findings", []) or []:
                # Add note metadata to the extracted finding
                f.update(
                    {
                        "note_type": note.get("note_type", "ECG"),
                        "charttime": note.get("charttime", ""),
                        "note_id": note.get("note_id"),
                    }
                )
                findings.append(f)
        return findings

    def _post_process_llm_findings(
        self, llm_findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and format findings extracted by the LLM."""
        validated_findings = []
        for finding in llm_findings:
            is_valid, validation_details = self._validate_finding(finding)
            if is_valid:
                validated_findings.append(
                    {
                        "finding": finding.get("finding"),
                        "description": finding.get("finding"),
                        "context": (finding.get("context") or "")[:600],
                        "leads": finding.get("leads") or [],
                        "measurements": finding.get("measurements") or {},
                        "is_new": bool(finding.get("is_new")),
                        "note_type": finding.get("note_type", "ECG"),
                        "charttime": finding.get("charttime", ""),
                        "note_id": finding.get("note_id"),
                        "mi_related": True,  # Assumed from LLM prompt
                        "validation": validation_details,
                    }
                )
        return validated_findings

    def _validate_finding(self, finding: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate a single ECG finding against clinical rules."""
        finding_name = finding.get("finding")
        leads = finding.get("leads", [])
        measurements = finding.get("measurements", {})
        is_new = finding.get("is_new", False)

        # Rule 1: Must be a new or presumably new finding for MI criteria
        if not is_new:
            return False, {"reason": "Finding is not new."}

        # Rule 2: Must be in at least two contiguous leads
        if not self._are_leads_contiguous(leads):
            return False, {"reason": f"Leads {leads} are not contiguous."}

        # Rule 3: Must meet numeric thresholds if applicable
        thresholds = self.VALIDATION_THRESHOLDS.get(finding_name)
        if thresholds:
            if finding_name == "ST Elevation":
                value = measurements.get("st_elevation_mm")
                if value is None or value < thresholds["mm"]:
                    return False, {
                        "reason": f'ST elevation of {value}mm is below threshold of {thresholds["mm"]}mm.'
                    }
            elif finding_name == "ST Depression":
                value = measurements.get("st_depression_mm")
                if value is None or value < thresholds["mm"]:
                    return False, {
                        "reason": f'ST depression of {value}mm is below threshold of {thresholds["mm"]}mm.'
                    }
            elif finding_name == "T Wave Inversion":
                value = measurements.get("t_wave_inversion_mm")
                if value is None or value < thresholds["mm"]:
                    return False, {
                        "reason": f'T wave inversion of {value}mm is below threshold of {thresholds["mm"]}mm.'
                    }

        return True, {"status": "Validated"}

    def _are_leads_contiguous(self, leads: List[str]) -> bool:
        """Check if at least two leads in the list are contiguous."""
        if not leads or len(leads) < 2:
            return False

        lead_set = set(leads)
        for group_name, group_leads in self.LEAD_GROUPS.items():
            # Count how many of the finding's leads are in this anatomical group
            intersection_count = len(lead_set.intersection(group_leads))
            if intersection_count >= 2:
                return True

        return False

    def _extract_findings_regex(self, notes: Any) -> List[Dict[str, Any]]:
        """Extract ECG findings from clinical notes using regex.

        Args:
            notes: DataFrame containing clinical notes

        Returns:
            List of ECG findings with context
        """
        findings = []
        processed_texts = set()

        for _, note in notes.iterrows():
            text = note.get("text", "")

            # Skip if we've already processed this text (duplicate notes)
            if text in processed_texts:
                continue

            processed_texts.add(text)

            # Check each ECG pattern
            for pattern_info in self.ECG_PATTERNS:
                pattern = pattern_info["pattern"]
                for match in pattern.finditer(text):
                    # Get context around the match
                    start = max(0, match.start() - 150)
                    end = min(len(text), match.end() + 150)
                    context = text[start:end].replace("\n", " ").strip()

                    # Clean up the context
                    context = "..." + context + "..."

                    # Extract leads if mentioned
                    leads = self._extract_leads(context)

                    # Extract measurements if mentioned
                    measurements = self._extract_measurements(context)

                    # Determine if new/acute
                    is_new = self._is_new_finding(context)

                    findings.append(
                        {
                            "finding": pattern_info["name"],
                            "description": match.group(0),
                            "context": context,
                            "leads": leads,
                            "measurements": measurements,
                            "is_new": is_new,
                            "note_type": note.get("note_type", "Unknown"),
                            "timestamp": note.get("charttime", ""),
                            "note_id": note.get("note_id"),
                            "mi_related": pattern_info["mi_related"],
                            "criteria": pattern_info["criteria"],
                        }
                    )

        return findings

    def _extract_leads(self, text: str) -> List[str]:
        """Extract ECG leads mentioned in the text.

        Args:
            text: Text containing ECG findings

        Returns:
            List of leads mentioned in the text
        """
        # Standard ECG leads
        standard_leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",  # Limb leads
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",  # Precordial leads
            "V7",
            "V8",
            "V9",  # Additional posterior leads
        ]

        # Look for lead groups first
        mentioned_leads = set()

        for group_name, leads in self.LEAD_GROUPS.items():
            if re.search(r"\b" + re.escape(group_name) + r"\b", text, re.IGNORECASE):
                mentioned_leads.update(leads)

        # Look for individual leads
        for lead in standard_leads:
            if re.search(r"\b" + re.escape(lead) + r"\b", text):
                mentioned_leads.add(lead)

        # Look for lead ranges (e.g., V1-V4)
        lead_ranges = re.finditer(r"([Vv]\s*\d+)\s*[-–]\s*([Vv]\s*\d+)", text)
        for match in lead_ranges:
            start_lead = match.group(1).upper().replace(" ", "")
            end_lead = match.group(2).upper().replace(" ", "")

            # Extract lead numbers
            start_num = int("".join(filter(str.isdigit, start_lead)))
            end_num = int("".join(filter(str.isdigit, end_lead)))

            # Add all leads in the range
            for num in range(start_num, end_num + 1):
                mentioned_leads.add(f"V{num}")

        return sorted(list(mentioned_leads))

    def _extract_measurements(self, text: str) -> Dict[str, Any]:
        """Extract measurements (e.g., ST elevation in mm) from text.

        Args:
            text: Text containing ECG measurements

        Returns:
            Dictionary of measurements with their values
        """
        measurements = {}

        # Look for ST elevation/ST depression measurements
        st_patterns = [
            (
                r"ST\s*[^a-z]*\s*(?:elevat|depress)[^\d]*([\d\.]+)\s*mm",
                "st_elevation_mm",
            ),
            (
                r"ST\s*[^a-z]*\s*(?:elevat|depress)[^\d]*([\d\.]+)\s*mV",
                "st_elevation_mv",
            ),
            (
                r"ST\s*[^a-z]*\s*([\d\.]+)\s*mm\s*(?:ST)?\s*(?:elevat|depress)",
                "st_elevation_mm",
            ),
            (
                r"ST\s*[^a-z]*\s*([\d\.]+)\s*mV\s*(?:ST)?\s*(?:elevat|depress)",
                "st_elevation_mv",
            ),
        ]

        for pattern, key in st_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    # Only keep the largest measurement if multiple are found
                    if key not in measurements or value > measurements[key]:
                        measurements[key] = value
                except (ValueError, IndexError):
                    continue

        # Look for Q wave duration
        q_wave_matches = re.finditer(
            r"Q\s*wave[^\d]*([\d\.]+)\s*(?:ms|msec|milliseconds?)", text, re.IGNORECASE
        )
        for match in q_wave_matches:
            try:
                value = float(match.group(1))
                measurements["q_wave_duration_ms"] = value
            except (ValueError, IndexError):
                continue

        return measurements

    def _is_new_finding(self, text: str) -> bool:
        """Determine if the finding is new/acute based on the context.

        Args:
            text: Text containing the finding

        Returns:
            True if the finding is new/acute, False otherwise
        """
        new_keywords = [
            "new",
            "acute",
            "recent",
            "new onset",
            "newly",
            "emerging",
            "developing",
            "evolving",
            "fresh",
            "just started",
            "just began",
            "just developed",
        ]

        old_keywords = [
            "old",
            "chronic",
            "resolving",
            "resolved",
            "previous",
            "prior",
            "history of",
            "h/o",
            "no new",
            "no acute",
            "no evidence of",
            "no sign of",
            "no indication of",
        ]

        # Check for new/acute indicators
        for keyword in new_keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
                return True

        # Check for old/chronic indicators
        for keyword in old_keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
                return False

        # Default to not new if no clear indication
        return False

    def _check_mi_criteria(self, ecg_findings: List[Dict]) -> Tuple[bool, Dict]:
        """Check if ECG findings meet MI criteria.

        Args:
            ecg_findings: List of ECG findings

        Returns:
            Tuple of (criteria_met, criteria_details)
        """
        if not ecg_findings:
            return False, {"reason": "No ECG findings available"}

        # Filter for MI-related findings that are new/acute
        mi_related = [
            f
            for f in ecg_findings
            if f.get("mi_related", False) and f.get("is_new", False)
        ]

        if not mi_related:
            return False, {"reason": "No new MI-related ECG findings"}

        # Check for specific MI criteria
        criteria_met = []

        # Check for ST elevation
        st_elevation = [f for f in mi_related if "ST Elevation" in f["finding"]]

        for finding in st_elevation:
            # Check if elevation is ≥1mm
            elevation_mm = finding.get("measurements", {}).get("st_elevation_mm", 0)
            elevation_mv = finding.get("measurements", {}).get("st_elevation_mv", 0)

            if elevation_mm >= 1.0 or elevation_mv >= 0.1:  # 0.1mV ≈ 1mm
                criteria_met.append(
                    {
                        "finding": finding["finding"],
                        "leads": finding.get("leads", []),
                        "elevation_mm": elevation_mm,
                        "criteria": finding.get("criteria", ""),
                    }
                )

        # Check for ST depression
        st_depression = [f for f in mi_related if "ST Depression" in f["finding"]]

        for finding in st_depression:
            # Check if depression is ≥0.5mm
            depression_mm = abs(
                finding.get("measurements", {}).get("st_elevation_mm", 0)
            )
            depression_mv = abs(
                finding.get("measurements", {}).get("st_elevation_mv", 0)
            )

            if depression_mm >= 0.5 or depression_mv >= 0.05:  # 0.05mV ≈ 0.5mm
                criteria_met.append(
                    {
                        "finding": finding["finding"],
                        "leads": finding.get("leads", []),
                        "depression_mm": depression_mm,
                        "criteria": finding.get("criteria", ""),
                    }
                )

        # Check for T wave inversion
        t_wave_inversion = [f for f in mi_related if "T Wave Inversion" in f["finding"]]

        for finding in t_wave_inversion:
            criteria_met.append(
                {
                    "finding": finding["finding"],
                    "leads": finding.get("leads", []),
                    "criteria": finding.get("criteria", ""),
                }
            )

        # Check for pathologic Q waves
        q_waves = [f for f in mi_related if "Pathologic Q Waves" in f["finding"]]

        for finding in q_waves:
            q_duration = finding.get("measurements", {}).get("q_wave_duration_ms", 0)
            criteria_met.append(
                {
                    "finding": finding["finding"],
                    "leads": finding.get("leads", []),
                    "q_wave_duration_ms": q_duration,
                    "criteria": finding.get("criteria", ""),
                }
            )

        # Check for LBBB
        lbbb = [f for f in mi_related if "Left Bundle Branch Block" in f["finding"]]

        if lbbb:
            criteria_met.append(
                {
                    "finding": "New Left Bundle Branch Block",
                    "criteria": "New or presumably new LBBB",
                    "note": "Considered diagnostic of MI in the appropriate clinical context",
                }
            )

        # Check for contiguous leads
        for finding in criteria_met:
            leads = finding.get("leads", [])
            if leads:
                # Check if any lead groups are fully represented
                for group_name, group_leads in self.LEAD_GROUPS.items():
                    if all(lead in leads for lead in group_leads):
                        finding["contiguous_leads"] = {
                            "group": group_name,
                            "leads": group_leads,
                        }
                        break

        # Determine if criteria are met
        mi_criteria_met = any(
            "contiguous_leads" in f
            or "Left Bundle Branch Block" in f.get("finding", "")
            for f in criteria_met
        )

        return mi_criteria_met, {"criteria_met": criteria_met}
