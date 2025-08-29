"""Evidence extractor for imaging findings related to MI."""
from typing import Dict, List, Any, Optional
import re
from .base_evidence_collector import BaseEvidenceCollector

class ImagingEvidenceExtractor(BaseEvidenceCollector):
    """Extracts imaging evidence (e.g., from echocardiograms) for MI."""

    IMAGING_PATTERNS = [
        {
            'name': 'Wall Motion Abnormality',
            'pattern': re.compile(r'(wall motion abnormalit(y|ies)|WMA|hypokinesis|akinesis|dyskinesis)', re.IGNORECASE),
            'mi_related': True
        }
    ]

    def __init__(self, notes_data_loader: Any, llm_client: Optional[Any] = None, max_notes: Optional[int] = None):
        """Initialize the imaging evidence extractor.

        Args:
            notes_data_loader: Instance of ClinicalNotesDataLoader
            llm_client: Optional instance of the LLM client
            max_notes: Maximum number of notes to process
        """
        super().__init__(notes_data_loader=notes_data_loader, llm_client=llm_client, max_notes=max_notes)

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect imaging evidence from notes for a specific admission."""
        evidence = self._get_evidence_base()
        evidence['imaging_findings'] = []
        evidence['metadata'] = {'extraction_mode': 'none'}

        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)
        # Filter for relevant notes like radiology, echo, etc.
        rad_notes = notes[notes['note_type'].isin(['Echo', 'Radiology'])]

        # Limit number of notes if configured
        if self.max_notes and len(rad_notes) > self.max_notes:
            rad_notes = rad_notes.head(self.max_notes)

        if rad_notes.empty:
            return evidence

        extracted_with_llm = False
        if self.llm_client and self.llm_client.enabled:
            try:
                findings = self._extract_with_llm(rad_notes)
                evidence['imaging_findings'] = findings
                evidence['metadata']['extraction_mode'] = 'llm'
                extracted_with_llm = True
            except Exception:
                pass

        if not extracted_with_llm:
            evidence['metadata']['extraction_mode'] = 'regex_fallback' if self.llm_client else 'regex'
            findings = self._extract_with_regex(rad_notes)
            evidence['imaging_findings'] = findings

        # Add a summary flag for the rule engine
        evidence['wall_motion_abnormalities'] = any(f['mi_related'] for f in evidence['imaging_findings'])

        return evidence

    def _extract_with_llm(self, notes: Any) -> List[Dict[str, Any]]:
        """Use LLM to extract imaging findings."""
        instructions = (
            "Read the imaging report and extract findings of new or presumed new loss of viable myocardium or regional "
            "wall motion abnormality. Return a JSON object with a key 'imaging_findings', a list of objects. "
            "Each object should have 'finding' (e.g., 'Wall Motion Abnormality'), 'context', and 'is_new' (boolean)."
        )
        findings = []
        for _, note in notes.iterrows():
            text = str(note.get('text', ''))
            if not text.strip():
                continue
            data = self.llm_client.extract_json(instructions, text)
            for f in data.get('imaging_findings', []) or []:
                if f.get('is_new'):
                    findings.append({
                        'finding': f.get('finding'),
                        'context': f.get('context'),
                        'is_new': True,
                        'mi_related': True,
                        'charttime': note.get('charttime')
                    })
        return findings

    def _extract_with_regex(self, notes: Any) -> List[Dict[str, Any]]:
        """Use regex to extract imaging findings."""
        findings = []
        for _, note in notes.iterrows():
            text = note.get('text', '')
            for pattern_info in self.IMAGING_PATTERNS:
                for match in pattern_info['pattern'].finditer(text):
                    # Simple check for 'new' or 'acute' in context
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    if re.search(r'\b(new|acute|worsening)\b', context, re.IGNORECASE):
                        findings.append({
                            'finding': pattern_info['name'],
                            'context': context.strip(),
                            'is_new': True,
                            'mi_related': True,
                            'charttime': note.get('charttime')
                        })
                        break # Move to next note after first new finding
        return findings
