"""Clinical evidence extractor for identifying MI-related symptoms and findings."""
from typing import Dict, List, Any, Set, Optional, Tuple
import re
from datetime import datetime
import logging
from .base_evidence_collector import BaseEvidenceCollector
from ..llm_client import LightenLLMClient

logger = logging.getLogger(__name__)

class ClinicalEvidenceExtractor(BaseEvidenceCollector):
    """Extracts clinical evidence of myocardial infarction from notes."""
    
    # Keywords for MI-related symptoms and findings (Criteria B.1 - Symptoms of myocardial ischemia)
    # Based on clinical guideline: chest pain, chest pressure, chest tightness, substernal discomfort, 
    # burning sensation, left arm/jaw/back pain (anginal equivalents), dyspnea, diaphoresis, nausea/vomiting
    SYMPTOM_KEYWORDS = [
        # Primary chest symptoms (per guideline)
        'chest pain', 'chest pressure', 'chest tightness',
        'substernal discomfort', 'substernal pain', 'substernal pressure',
        'burning sensation', 'chest burning', 'chest heaviness', 'angina',
        
        # Anginal equivalents (per guideline)
        'left arm pain', 'jaw pain', 'back pain',
        'radiat.* arm', 'radiat.* jaw', 'radiat.* neck', 
        'radiat.* back', 'radiat.* shoulder',
        
        # Associated symptoms in appropriate clinical context (per guideline)
        'dyspnea', 'shortness of breath', 'sob', 'diaphoresis',
        'sweating', 'nausea', 'vomiting', 'lightheaded', 'dizzy',
        'syncope', 'palpitations', 'fatigue', 'weakness',
        'indigestion', 'heartburn', 'epigastric pain',
        
        # Additional clinical descriptors
        'crushing chest pain', 'squeezing chest pain', 'pressure-like pain'
    ]
    
    # Regular expressions for symptom patterns
    SYMPTOM_PATTERNS = [
        {
            'name': 'Chest Pain',
            'pattern': re.compile(r'chest\s+(?:pain|discomfort|pressure|tightness|heaviness|burning)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Substernal Pain',
            'pattern': re.compile(r'substernal\s+(?:pain|discomfort|pressure|tightness)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Radiation to Arm',
            'pattern': re.compile(r'radiat(?:e|ing|es|ed).*?(?:arm)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Radiation to Jaw',
            'pattern': re.compile(r'radiat(?:e|ing|es|ed).*?(?:jaw)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Radiation to Neck',
            'pattern': re.compile(r'radiat(?:e|ing|es|ed).*?(?:neck)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Radiation to Back',
            'pattern': re.compile(r'radiat(?:e|ing|es|ed).*?(?:back)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Radiation to Shoulder',
            'pattern': re.compile(r'radiat(?:e|ing|es|ed).*?(?:shoulder)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Shortness of Breath',
            'pattern': re.compile(r'(?:shortness of breath|sob|dyspnea)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Diaphoresis',
            'pattern': re.compile(r'(?:diaphoresis|sweating)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Nausea/Vomiting',
            'pattern': re.compile(r'(?:nausea|vomiting)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Lightheaded/Dizzy',
            'pattern': re.compile(r'(?:lightheaded|dizzy|syncope)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Palpitations',
            'pattern': re.compile(r'(?:palpitations|heart racing|pounding)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Fatigue/Weakness',
            'pattern': re.compile(r'(?:fatigue|weakness|tired)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Indigestion/Heartburn',
            'pattern': re.compile(r'(?:indigestion|heartburn|epigastric pain)', re.IGNORECASE),
            'mi_related': True
        },
        {
            'name': 'Atypical Chest Pain',
            'pattern': re.compile(r'atypical\s+chest\s+pain', re.IGNORECASE),
            'mi_related': False
        }
    ]

    DIAGNOSIS_PATTERNS = [
        {
            'name': 'MI Diagnosis',
            'pattern': re.compile(r'\b(myocardial infarction|mi|n?stemi)\b', re.IGNORECASE),
        }
    ]

    # Keywords for MI-related findings
    FINDING_KEYWORDS = [
        'acute coronary syndrome', 'acs', 'stemi', 'nstemi',
        'myocardial infarction', 'heart attack', 'cardiac ischemia',
        r'ejection fraction', r'ef\s*[<]?\s*40%?', r'cardiogenic shock'
    ]
    
    def __init__(self, notes_data_loader: Any, llm_client: Optional[Any] = None, max_notes: Optional[int] = None):
        """Initialize the clinical evidence extractor.
        
        Args:
            notes_data_loader: Instance of ClinicalNotesDataLoader for accessing clinical notes
            llm_client: Optional instance of the LLM client
            max_notes: Maximum number of notes to process
        """
        super().__init__(notes_data_loader=notes_data_loader, llm_client=llm_client, max_notes=max_notes)
    
    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect clinical evidence from notes for a specific admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing clinical evidence
        """
        evidence = self._get_evidence_base()
        evidence['symptoms'] = []
        evidence['diagnoses'] = []
        evidence['metadata'] = {'extraction_mode': 'none'}

        # Get clinical notes for the admission
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)

        # Limit number of notes if configured
        if self.max_notes and len(notes) > self.max_notes:
            notes = notes.head(self.max_notes)

        if notes.empty:
            return evidence

        # If LLM client is available, use it for extraction
        try_llm = getattr(self, 'llm_client', None) and isinstance(self.llm_client, LightenLLMClient) and self.llm_client.enabled
        if try_llm:
            try:
                symptoms, diagnoses = self._extract_with_llm(notes)
                mode = 'llm'
            except Exception:
                symptoms, diagnoses = self._extract_with_regex(notes)
                mode = 'regex_fallback'
        else:
            symptoms, diagnoses = self._extract_with_regex(notes)
            mode = 'regex'
        
        evidence.update({
            'symptoms': symptoms,
            'diagnoses': diagnoses,
            'sources': [{
                'type': 'clinical_notes',
                'symptom_count': len(symptoms),
                'diagnosis_count': len(diagnoses)
            }],
            'metadata': {**evidence.get('metadata', {}), 'extraction_mode': mode}
        })
        
        # Add summary flags for the rule engine
        evidence['ischemic_symptoms_present'] = any(s['mi_related'] for s in evidence['symptoms'])

        return evidence
    
    def _extract_with_llm(self, notes: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Use LLM to extract symptoms and diagnoses."""
        instructions = (
            "Read the clinical note and extract two types of information in a single JSON object: "
            "1. A list of ischemic symptoms under the key 'symptoms'. Each object should have 'symptom', 'context', and 'onset_time'. "
            "2. A list of MI diagnoses under the key 'diagnoses'. Each object should have 'diagnosis' and 'diagnosis_date'."
        )
        symptoms, diagnoses = [], []
        for _, note in notes.iterrows():
            text = str(note.get('text', ''))
            if not text.strip():
                continue
            data = self.llm_client.extract_json(instructions, text)
            # Process symptoms
            for s in data.get('symptoms', []) or []:
                symptoms.append({
                    'symptom': s.get('symptom'),
                    'context': s.get('context'),
                    'onset_time': s.get('onset_time'),
                    'charttime': note.get('charttime')
                })
            # Process diagnoses
            for d in data.get('diagnoses', []) or []:
                diagnoses.append({
                    'diagnosis': d.get('diagnosis'),
                    'diagnosis_date': d.get('diagnosis_date'),
                    'charttime': note.get('charttime')
                })
        return symptoms, diagnoses

    def _extract_with_regex(self, notes: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Use regex to extract symptoms and diagnoses."""
        symptoms, diagnoses = [], []
        for _, note in notes.iterrows():
            text = note.get('text', '')
            charttime = note.get('charttime')

            # Extract symptoms
            for pattern_info in self.SYMPTOM_PATTERNS:
                for match in pattern_info['pattern'].finditer(text):
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    symptoms.append({
                        'symptom': pattern_info['name'],
                        'context': context.strip(),
                        'mi_related': pattern_info['mi_related'],
                        'charttime': charttime
                    })

            # Extract diagnoses
            for pattern_info in self.DIAGNOSIS_PATTERNS:
                for match in pattern_info['pattern'].finditer(text):
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    diagnoses.append({
                        'diagnosis': pattern_info['name'],
                        'context': context.strip(),
                        'diagnosis_date': charttime, # Use note time as proxy
                        'charttime': charttime
                    })

        return symptoms, diagnoses

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
