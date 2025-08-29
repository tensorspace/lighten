"""ECG evidence extractor for identifying MI-related ECG findings."""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from .base_evidence_collector import BaseEvidenceCollector
from ..llm_client import LightenLLMClient

class ECGEvidenceExtractor(BaseEvidenceCollector):
    """Extracts ECG evidence related to myocardial infarction."""
    
    # ECG patterns indicative of MI
    ECG_PATTERNS = [
        # ST elevation
        {
            'name': 'ST Elevation',
            'pattern': re.compile(r'ST\s*(?:segment\s*)?elevat', re.IGNORECASE),
            'criteria': 'New ST elevation ≥1mm in two contiguous leads',
            'mi_related': True
        },
        # ST depression
        {
            'name': 'ST Depression',
            'pattern': re.compile(r'ST\s*(?:segment\s*)?depress', re.IGNORECASE),
            'criteria': 'New horizontal or down-sloping ST depression ≥0.5mm in two contiguous leads',
            'mi_related': True
        },
        # T wave inversion
        {
            'name': 'T Wave Inversion',
            'pattern': re.compile(r'T\s*wave\s*inver', re.IGNORECASE),
            'criteria': 'New T wave inversion ≥1mm in two contiguous leads with prominent R wave or R/S ratio >1',
            'mi_related': True
        },
        # Pathologic Q waves
        {
            'name': 'Pathologic Q Waves',
            'pattern': re.compile(r'(?:new|pathologic)\s*q\s*waves?', re.IGNORECASE),
            'criteria': 'New Q waves ≥0.02 seconds or QS complex in ≥2 contiguous leads',
            'mi_related': True
        },
        # Other ECG findings
        {
            'name': 'Left Bundle Branch Block',
            'pattern': re.compile(r'left\s+bundle\s+branch\s+block|LBBB', re.IGNORECASE),
            'criteria': 'New or presumably new LBBB',
            'mi_related': True
        },
        {
            'name': 'Ventricular Arrhythmia',
            'pattern': re.compile(r'ventricular\s+(?:tachycardia|fibrillation)|VT\/VF', re.IGNORECASE),
            'criteria': 'Ventricular tachycardia or fibrillation',
            'mi_related': True
        },
        {
            'name': 'Atrial Fibrillation',
            'pattern': re.compile(r'atrial\s+fibrillation|a\s*?fib', re.IGNORECASE),
            'criteria': 'New onset atrial fibrillation',
            'mi_related': False
        }
    ]
    
    # Lead groups for contiguous leads
    LEAD_GROUPS = {
        'inferior': ['II', 'III', 'aVF'],
        'lateral': ['I', 'aVL', 'V5', 'V6'],
        'anterior': ['V1', 'V2', 'V3', 'V4'],
        'septal': ['V1', 'V2'],
        'anteroseptal': ['V1', 'V2', 'V3', 'V4'],
        'anterolateral': ['I', 'aVL', 'V3', 'V4', 'V5', 'V6'],
        'inferolateral': ['II', 'III', 'aVF', 'V5', 'V6'],
        'inferoposterior': ['II', 'III', 'aVF', 'V7', 'V8', 'V9']
    }
    
    def __init__(self, notes_loader: Any, llm_client: Any = None):
        """Initialize the ECG evidence extractor.
        
        Args:
            notes_loader: Instance of ClinicalNotesLoader for accessing clinical notes
        """
        super().__init__(notes_loader=notes_loader, llm_client=llm_client)
    
    def collect_evidence(self, patient_id: str) -> Dict[str, Any]:
        """Collect ECG evidence for a patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing ECG evidence
        """
        evidence = self._get_evidence_base()
        
        if not self.notes_loader:
            evidence['error'] = 'Notes loader not provided'
            return evidence
        
        # Get notes
        notes = self.notes_loader.get_patient_notes(patient_id)
        if notes.empty:
            evidence['ecg_findings'] = []
            evidence['metadata'] = {**evidence.get('metadata', {}), 'extraction_mode': 'none_no_notes'}
            return evidence
        
        # LLM-first with regex fallback
        try_llm = getattr(self, 'llm_client', None) and isinstance(self.llm_client, LightenLLMClient) and self.llm_client.enabled
        if try_llm:
            try:
                ecg_findings = self._extract_with_llm(notes)
                mode = 'llm'
            except Exception:
                ecg_findings = self._extract_ecg_findings(notes)
                mode = 'regex_fallback'
        else:
            ecg_findings = self._extract_ecg_findings(notes)
            mode = 'regex'
        
        # Determine if findings meet MI criteria
        mi_criteria_met, criteria_details = self._check_mi_criteria(ecg_findings)
        
        evidence.update({
            'ecg_findings': ecg_findings,
            'mi_criteria_met': mi_criteria_met,
            'criteria_details': criteria_details,
            'sources': [{
                'type': 'clinical_notes',
                'ecg_finding_count': len(ecg_findings)
            }],
            'metadata': {**evidence.get('metadata', {}), 'extraction_mode': mode}
        })
        
        return evidence

    def _extract_with_llm(self, notes: Any) -> List[Dict[str, Any]]:
        """Use LLM to extract ECG findings structured for MI criteria."""
        instructions = (
            "Read the clinical text and extract ECG-related findings relevant to myocardial infarction. "
            "Return JSON with key 'ecg_findings' (array of objects) where each object has: "
            "{finding: string, context: string, leads: array of strings, measurements: object, is_new: boolean}. "
            "Leads should be like I, II, III, aVR, aVL, aVF, V1..V9. "
            "Measurements may include: st_elevation_mm, st_elevation_mv, q_wave_duration_ms."
        )
        findings: List[Dict[str, Any]] = []
        for _, note in notes.iterrows():
            text = str(note.get('text', ''))
            if not text.strip():
                continue
            data = self.llm_client.extract_json(instructions, text)
            for f in data.get('ecg_findings', []) or []:
                findings.append({
                    'finding': f.get('finding'),
                    'description': f.get('finding'),
                    'context': (f.get('context') or '')[:600],
                    'leads': f.get('leads') or [],
                    'measurements': f.get('measurements') or {},
                    'is_new': bool(f.get('is_new')),
                    'note_type': note.get('note_type', 'Unknown'),
                    'timestamp': note.get('charttime', ''),
                    'note_id': note.get('note_id'),
                    'mi_related': True,
                    'criteria': ''
                })
        return findings
        
        # Get all notes for the patient
        notes = self.notes_loader.get_patient_notes(patient_id)
        if notes.empty:
            evidence['ecg_findings'] = []
            return evidence
        
        # Extract ECG findings (LLM first if available)
        if getattr(self, 'llm_client', None) and isinstance(self.llm_client, LightenLLMClient) and self.llm_client.enabled:
            try:
                ecg_findings = self._extract_with_llm(notes)
                evidence['metadata']['extraction_mode'] = 'llm'
            except Exception:
                ecg_findings = self._extract_ecg_findings(notes)
                evidence['metadata']['extraction_mode'] = 'regex_fallback'
        else:
            ecg_findings = self._extract_ecg_findings(notes)
            evidence['metadata']['extraction_mode'] = 'regex'
        
        # Determine if findings meet MI criteria
        mi_criteria_met, criteria_details = self._check_mi_criteria(ecg_findings)
        
        evidence.update({
            'ecg_findings': ecg_findings,
            'mi_criteria_met': mi_criteria_met,
            'criteria_details': criteria_details,
            'sources': [{
                'type': 'clinical_notes',
                'ecg_finding_count': len(ecg_findings)
            }]
        })
        
        return evidence
    
    def _extract_ecg_findings(self, notes: Any) -> List[Dict[str, Any]]:
        """Extract ECG findings from clinical notes.
        
        Args:
            notes: DataFrame containing clinical notes
            
        Returns:
            List of ECG findings with context
        """
        findings = []
        processed_texts = set()
        
        for _, note in notes.iterrows():
            text = note.get('text', '')
            
            # Skip if we've already processed this text (duplicate notes)
            if text in processed_texts:
                continue
                
            processed_texts.add(text)
            
            # Check each ECG pattern
            for pattern_info in self.ECG_PATTERNS:
                pattern = pattern_info['pattern']
                for match in pattern.finditer(text):
                    # Get context around the match
                    start = max(0, match.start() - 150)
                    end = min(len(text), match.end() + 150)
                    context = text[start:end].replace('\n', ' ').strip()
                    
                    # Clean up the context
                    context = '...' + context + '...'
                    
                    # Extract leads if mentioned
                    leads = self._extract_leads(context)
                    
                    # Extract measurements if mentioned
                    measurements = self._extract_measurements(context)
                    
                    # Determine if new/acute
                    is_new = self._is_new_finding(context)
                    
                    findings.append({
                        'finding': pattern_info['name'],
                        'description': match.group(0),
                        'context': context,
                        'leads': leads,
                        'measurements': measurements,
                        'is_new': is_new,
                        'note_type': note.get('note_type', 'Unknown'),
                        'timestamp': note.get('charttime', ''),
                        'note_id': note.get('note_id'),
                        'mi_related': pattern_info['mi_related'],
                        'criteria': pattern_info['criteria']
                    })
        
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
            'I', 'II', 'III', 'aVR', 'aVL', 'aVF',  # Limb leads
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6',    # Precordial leads
            'V7', 'V8', 'V9'                        # Additional posterior leads
        ]
        
        # Look for lead groups first
        mentioned_leads = set()
        
        for group_name, leads in self.LEAD_GROUPS.items():
            if re.search(r'\b' + re.escape(group_name) + r'\b', text, re.IGNORECASE):
                mentioned_leads.update(leads)
        
        # Look for individual leads
        for lead in standard_leads:
            if re.search(r'\b' + re.escape(lead) + r'\b', text):
                mentioned_leads.add(lead)
        
        # Look for lead ranges (e.g., V1-V4)
        lead_ranges = re.finditer(r'([Vv]\s*\d+)\s*[-–]\s*([Vv]\s*\d+)', text)
        for match in lead_ranges:
            start_lead = match.group(1).upper().replace(' ', '')
            end_lead = match.group(2).upper().replace(' ', '')
            
            # Extract lead numbers
            start_num = int(''.join(filter(str.isdigit, start_lead)))
            end_num = int(''.join(filter(str.isdigit, end_lead)))
            
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
            (r'ST\s*[^a-z]*\s*(?:elevat|depress)[^\d]*([\d\.]+)\s*mm', 'st_elevation_mm'),
            (r'ST\s*[^a-z]*\s*(?:elevat|depress)[^\d]*([\d\.]+)\s*mV', 'st_elevation_mv'),
            (r'ST\s*[^a-z]*\s*([\d\.]+)\s*mm\s*(?:ST)?\s*(?:elevat|depress)', 'st_elevation_mm'),
            (r'ST\s*[^a-z]*\s*([\d\.]+)\s*mV\s*(?:ST)?\s*(?:elevat|depress)', 'st_elevation_mv')
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
        q_wave_matches = re.finditer(r'Q\s*wave[^\d]*([\d\.]+)\s*(?:ms|msec|milliseconds?)', text, re.IGNORECASE)
        for match in q_wave_matches:
            try:
                value = float(match.group(1))
                measurements['q_wave_duration_ms'] = value
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
            'new', 'acute', 'recent', 'new onset', 'newly',
            'emerging', 'developing', 'evolving', 'fresh',
            'just started', 'just began', 'just developed'
        ]
        
        old_keywords = [
            'old', 'chronic', 'resolving', 'resolved',
            'previous', 'prior', 'history of', 'h\/o',
            'no new', 'no acute', 'no evidence of',
            'no sign of', 'no indication of'
        ]
        
        # Check for new/acute indicators
        for keyword in new_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                return True
        
        # Check for old/chronic indicators
        for keyword in old_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
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
            return False, {'reason': 'No ECG findings available'}
        
        # Filter for MI-related findings that are new/acute
        mi_related = [f for f in ecg_findings 
                     if f.get('mi_related', False) and f.get('is_new', False)]
        
        if not mi_related:
            return False, {'reason': 'No new MI-related ECG findings'}
        
        # Check for specific MI criteria
        criteria_met = []
        
        # Check for ST elevation
        st_elevation = [f for f in mi_related 
                       if 'ST Elevation' in f['finding']]
        
        for finding in st_elevation:
            # Check if elevation is ≥1mm
            elevation_mm = finding.get('measurements', {}).get('st_elevation_mm', 0)
            elevation_mv = finding.get('measurements', {}).get('st_elevation_mv', 0)
            
            if elevation_mm >= 1.0 or elevation_mv >= 0.1:  # 0.1mV ≈ 1mm
                criteria_met.append({
                    'finding': finding['finding'],
                    'leads': finding.get('leads', []),
                    'elevation_mm': elevation_mm,
                    'criteria': finding.get('criteria', '')
                })
        
        # Check for ST depression
        st_depression = [f for f in mi_related 
                        if 'ST Depression' in f['finding']]
        
        for finding in st_depression:
            # Check if depression is ≥0.5mm
            depression_mm = abs(finding.get('measurements', {}).get('st_elevation_mm', 0))
            depression_mv = abs(finding.get('measurements', {}).get('st_elevation_mv', 0))
            
            if depression_mm >= 0.5 or depression_mv >= 0.05:  # 0.05mV ≈ 0.5mm
                criteria_met.append({
                    'finding': finding['finding'],
                    'leads': finding.get('leads', []),
                    'depression_mm': depression_mm,
                    'criteria': finding.get('criteria', '')
                })
        
        # Check for T wave inversion
        t_wave_inversion = [f for f in mi_related 
                           if 'T Wave Inversion' in f['finding']]
        
        for finding in t_wave_inversion:
            criteria_met.append({
                'finding': finding['finding'],
                'leads': finding.get('leads', []),
                'criteria': finding.get('criteria', '')
            })
        
        # Check for pathologic Q waves
        q_waves = [f for f in mi_related 
                  if 'Pathologic Q Waves' in f['finding']]
        
        for finding in q_waves:
            q_duration = finding.get('measurements', {}).get('q_wave_duration_ms', 0)
            criteria_met.append({
                'finding': finding['finding'],
                'leads': finding.get('leads', []),
                'q_wave_duration_ms': q_duration,
                'criteria': finding.get('criteria', '')
            })
        
        # Check for LBBB
        lbbb = [f for f in mi_related 
               if 'Left Bundle Branch Block' in f['finding']]
        
        if lbbb:
            criteria_met.append({
                'finding': 'New Left Bundle Branch Block',
                'criteria': 'New or presumably new LBBB',
                'note': 'Considered diagnostic of MI in the appropriate clinical context'
            })
        
        # Check for contiguous leads
        for finding in criteria_met:
            leads = finding.get('leads', [])
            if leads:
                # Check if any lead groups are fully represented
                for group_name, group_leads in self.LEAD_GROUPS.items():
                    if all(lead in leads for lead in group_leads):
                        finding['contiguous_leads'] = {
                            'group': group_name,
                            'leads': group_leads
                        }
                        break
        
        # Determine if criteria are met
        mi_criteria_met = any(
            'contiguous_leads' in f or 
            'Left Bundle Branch Block' in f.get('finding', '') 
            for f in criteria_met
        )
        
        return mi_criteria_met, {'criteria_met': criteria_met}
