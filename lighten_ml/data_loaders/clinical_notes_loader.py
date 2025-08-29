"""Clinical notes loading and processing."""
from typing import Dict, List, Any, Optional
import pandas as pd
from .base_loader import BaseDataLoader

class ClinicalNotesLoader(BaseDataLoader):
    """Load and process clinical notes."""
    
    def __init__(self, notes_path: str):
        """Initialize the clinical notes loader.
        
        Args:
            notes_path: Path to the clinical notes CSV file
        """
        super().__init__(notes_path)
        self.notes_columns = ['note_id', 'subject_id', 'hadm_id', 'note_type', 'charttime', 'text']
    
    def load_data(self) -> None:
        """Load and preprocess clinical notes."""
        # Load notes data
        self.data = pd.read_csv(
            self.file_path,
            usecols=lambda x: x in self.notes_columns,
            parse_dates=['charttime'],
            infer_datetime_format=True
        )
        
        # Clean text data
        if 'text' in self.data.columns:
            self.data['text'] = self.data['text'].fillna('').astype(str)
    
    def get_patient_notes(self, patient_id: str) -> pd.DataFrame:
        """Get all clinical notes for a specific patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            DataFrame containing the patient's clinical notes
        """
        if self.data is None:
            self.load_data()
            
        patient_notes = self.data[self.data['subject_id'] == patient_id].copy()
        
        # Sort by charttime
        if not patient_notes.empty and 'charttime' in patient_notes.columns:
            patient_notes = patient_notes.sort_values('charttime')
            
        return patient_notes
    
    def search_notes_for_keywords(self, patient_id: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search patient's notes for specific keywords.
        
        Args:
            patient_id: The ID of the patient
            keywords: List of keywords to search for
            
        Returns:
            List of dictionaries containing matching notes and context
        """
        notes = self.get_patient_notes(patient_id)
        if notes.empty:
            return []
        
        results = []
        
        for _, note in notes.iterrows():
            text = note.get('text', '').lower()
            note_matches = []
            
            for keyword in keywords:
                if keyword.lower() in text:
                    # Find the context around the keyword
                    start = max(0, text.find(keyword.lower()) - 100)
                    end = min(len(text), text.find(keyword.lower()) + len(keyword) + 100)
                    context = text[start:end].replace('\n', ' ').strip()
                    note_matches.append({
                        'keyword': keyword,
                        'context': context,
                        'charttime': note.get('charttime'),
                        'note_type': note.get('note_type')
                    })
            
            if note_matches:
                results.append({
                    'note_id': note.get('note_id'),
                    'charttime': note.get('charttime'),
                    'note_type': note.get('note_type'),
                    'matches': note_matches
                })
        
        return results
    
    def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get clinical notes data for a specific patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing the patient's clinical notes data
        """
        notes = self.get_patient_notes(patient_id)
        return {
            'notes': notes.to_dict('records') if not notes.empty else []
        }
