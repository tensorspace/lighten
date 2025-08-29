"""Clinical evidence extractor for identifying MI-related symptoms and findings."""
from typing import Dict, List, Any, Set, Optional
import re
from datetime import datetime
from .base_evidence_collector import BaseEvidenceCollector

class ClinicalEvidenceExtractor(BaseEvidenceCollector):
    """Extracts clinical evidence of myocardial infarction from notes."""
    
    # Keywords for MI-related symptoms and findings
    SYMPTOM_KEYWORDS = [
        # Chest pain/discomfort
        'chest pain', 'chest pressure', 'chest tightness',
        'substernal pain', 'substernal pressure', 'substernal discomfort',
        'chest heaviness', 'chest burning', 'angina',
        
        # Radiation patterns
        'radiat.* arm', 'radiat.* jaw', 'radiat.* neck', 
        'radiat.* back', 'radiat.* shoulder',
        
        # Associated symptoms
        'dyspnea', 'shortness of breath', 'sob', 'diaphoresis',
        'sweating', 'nausea', 'vomiting', 'lightheaded', 'dizzy',
        'syncope', 'palpitations', 'fatigue', 'weakness',
        'indigestion', 'heartburn', 'epigastric pain'
    ]
    
    # Regular expressions for symptom patterns
    SYMPTOM_PATTERNS = [
        re.compile(r'chest\s+(?:pain|discomfort|pressure|tightness|heaviness|burning)', re.IGNORECASE),
        re.compile(r'substernal\s+(?:pain|discomfort|pressure|tightness)', re.IGNORECASE),
        re.compile(r'radiat(?:e|ing|es|ed).*?(?:arm|jaw|neck|back|shoulder)', re.IGNORECASE),
        re.compile(r'(?:shortness of breath|sob|dyspnea)', re.IGNORECASE),
        re.compile(r'(?:diaphoresis|sweating)', re.IGNORECASE),
        re.compile(r'(?:nausea|vomiting)', re.IGNORECASE),
        re.compile(r'(?:lightheaded|dizzy|syncope)', re.IGNORECASE),
        re.compile(r'(?:palpitations|heart racing|pounding)', re.IGNORECASE),
        re.compile(r'(?:fatigue|weakness|tired)', re.IGNORECASE),
        re.compile(r'(?:indigestion|heartburn|epigastric pain)', re.IGNORECASE)
    ]
    
    # Keywords for MI-related findings
    FINDING_KEYWORDS = [
        'acute coronary syndrome', 'acs', 'stemi', 'nstemi',
        'myocardial infarction', 'heart attack', 'cardiac ischemia',
        'elevated troponin', 'positive troponin', 'troponin elevation',
        'st elevation', 'st depression', 't wave inversion',
        'new q waves', 'pathologic q waves', 'new st changes',
        'wall motion abnormality', 'hypokinesis', 'akinesis',
        'ejection fraction', 'ef\s*[<]?\s*40%?', 'cardiogenic shock'
    ]
    
    def __init__(self, notes_loader: Any):
        """Initialize the clinical evidence extractor.
        
        Args:
            notes_loader: Instance of ClinicalNotesLoader for accessing clinical notes
        """
        super().__init__(notes_loader=notes_loader)
        self.compiled_keywords = [re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE) 
                                for kw in self.SYMPTOM_KEYWORDS + self.FINDING_KEYWORDS]
    
    def collect_evidence(self, patient_id: str) -> Dict[str, Any]:
        """Collect clinical evidence for a patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing clinical evidence
        """
        evidence = self._get_evidence_base()
        
        if not self.notes_loader:
            evidence['error'] = 'Notes loader not provided'
            return evidence
        
        # Search for symptoms and findings in notes
        symptoms = self._extract_symptoms(patient_id)
        findings = self._extract_findings(patient_id)
        
        evidence.update({
            'symptoms': symptoms,
            'findings': findings,
            'sources': [{
                'type': 'clinical_notes',
                'symptom_count': len(symptoms),
                'finding_count': len(findings)
            }]
        })
        
        return evidence
    
    def _extract_symptoms(self, patient_id: str) -> List[Dict[str, Any]]:
        """Extract MI-related symptoms from clinical notes.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            List of symptom findings with context
        """
        if not self.notes_loader:
            return []
        
        # Get all notes for the patient
        notes = self.notes_loader.get_patient_notes(patient_id)
        if notes.empty:
            return []
        
        symptoms = []
        processed_texts = set()  # To avoid duplicate processing
        
        for _, note in notes.iterrows():
            text = note.get('text', '').lower()
            
            # Skip if we've already processed this text (duplicate notes)
            if text in processed_texts:
                continue
                
            processed_texts.add(text)
            
            # Check each symptom pattern
            for pattern in self.SYMPTOM_PATTERNS:
                for match in pattern.finditer(text):
                    # Get context around the match
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].replace('\n', ' ').strip()
                    
                    # Clean up the context
                    context = '...' + context + '...'
                    
                    symptoms.append({
                        'symptom': match.group(0).strip(),
                        'context': context,
                        'note_type': note.get('note_type', 'Unknown'),
                        'timestamp': note.get('charttime', ''),
                        'note_id': note.get('note_id')
                    })
        
        return symptoms
    
    def _extract_findings(self, patient_id: str) -> List[Dict[str, Any]]:
        """Extract MI-related findings from clinical notes.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            List of clinical findings with context
        """
        if not self.notes_loader:
            return []
        
        # Get all notes for the patient
        notes = self.notes_loader.get_patient_notes(patient_id)
        if notes.empty:
            return []
        
        findings = []
        processed_texts = set()  # To avoid duplicate processing
        
        # Compile regex patterns for findings
        finding_patterns = [
            re.compile(r'(?:acute coronary syndrome|acs)', re.IGNORECASE),
            re.compile(r'(?:st\s*elevation|stemi)', re.IGNORECASE),
            re.compile(r'(?:non-?st\s*elevation|nstemi)', re.IGNORECASE),
            re.compile(r'myocardial\s+infarct', re.IGNORECASE),
            re.compile(r'heart\s+attack', re.IGNORECASE),
            re.compile(r'cardiac\s+ischemi', re.IGNORECASE),
            re.compile(r'(?:elevated|positive|elevation of)\s+troponin', re.IGNORECASE),
            re.compile(r'st\s*[^a-z]\s*elevat', re.IGNORECASE),
            re.compile(r'st\s*[^a-z]\s*depress', re.IGNORECASE),
            re.compile(r't\s*wave\s*inver', re.IGNORECASE),
            re.compile(r'(?:new|pathologic)\s*q\s*waves?', re.IGNORECASE),
            re.compile(r'wall\s*motion\s*(?:abnormal|hypokinesis|akinesis)', re.IGNORECASE),
            re.compile(r'ejection\s*fraction\s*(?:of|is)?\s*[<]?\s*\d{1,2}%?', re.IGNORECASE),
            re.compile(r'ef\s*[<]?\s*\d{1,2}%?', re.IGNORECASE),
            re.compile(r'cardiogenic\s*shock', re.IGNORECASE)
        ]
        
        for _, note in notes.iterrows():
            text = note.get('text', '').lower()
            
            # Skip if we've already processed this text (duplicate notes)
            if text in processed_texts:
                continue
                
            processed_texts.add(text)
            
            # Check each finding pattern
            for pattern in finding_patterns:
                for match in pattern.finditer(text):
                    # Get context around the match
                    start = max(0, match.start() - 150)
                    end = min(len(text), match.end() + 150)
                    context = text[start:end].replace('\n', ' ').strip()
                    
                    # Clean up the context
                    context = '...' + context + '...'
                    
                    findings.append({
                        'finding': match.group(0).strip(),
                        'context': context,
                        'note_type': note.get('note_type', 'Unknown'),
                        'timestamp': note.get('charttime', ''),
                        'note_id': note.get('note_id')
                    })
        
        return findings
    
    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from clinical text.
        
        Args:
            text: Clinical note text
            
        Returns:
            Dictionary with temporal information
        """
        # This is a simplified version - a more robust implementation would use NLP
        temporal_terms = {
            'acute': ['acute', 'sudden', 'abrupt', 'rapid'],
            'subacute': ['subacute', 'recent', 'past few', 'last few'],
            'chronic': ['chronic', 'long-standing', 'persistent', 'ongoing']
        }
        
        time_units = {
            'minutes': ['minute', 'min', 'mins'],
            'hours': ['hour', 'hr', 'hrs'],
            'days': ['day', 'days'],
            'weeks': ['week', 'wk', 'wks'],
            'months': ['month', 'mo', 'mos']
        }
        
        result = {
            'temporal_quality': 'unknown',
            'duration': None,
            'unit': None
        }
        
        # Check for temporal quality
        text_lower = text.lower()
        for quality, terms in temporal_terms.items():
            if any(term in text_lower for term in terms):
                result['temporal_quality'] = quality
                break
        
        # Look for duration patterns (e.g., "for 2 days", "since yesterday")
        duration_patterns = [
            r'(?:for|since|over|about|approximately|~)?\s*(\d+)\s*([a-z]+)',
            r'(?:for|since|over|about|approximately|~)?\s*(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+)'
        ]
        
        for pattern in duration_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    try:
                        # Try to convert to number
                        try:
                            value = int(match.group(1))
                        except (ValueError, IndexError):
                            # Handle word numbers
                            word_nums = {
                                'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3,
                                'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                                'eight': 8, 'nine': 9, 'ten': 10
                            }
                            value = word_nums.get(match.group(1).lower(), 1)
                        
                        # Get the unit
                        unit = match.group(2).lower()
                        
                        # Map to standard units
                        for std_unit, variants in time_units.items():
                            if any(variant in unit for variant in variants):
                                result.update({
                                    'duration': value,
                                    'unit': std_unit
                                })
                                break
                                
                    except (IndexError, AttributeError):
                        continue
        
        return result
