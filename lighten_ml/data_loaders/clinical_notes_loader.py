"""Clinical notes loading and processing."""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from .base_loader import BaseDataLoader
import logging

logger = logging.getLogger(__name__)

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
        if self.data is not None:
            return

        logger.info(f"Loading clinical notes from {self.file_path}...")
        try:
            # Load notes data
            self.data = pd.read_csv(
                self.file_path,
                usecols=lambda x: x in self.notes_columns,
                parse_dates=['charttime'],
                dtype={'subject_id': str, 'hadm_id': str}
            )

            # Clean text data
            if 'text' in self.data.columns:
                self.data['text'] = self.data['text'].fillna('').astype(str)

            logger.info("Clinical notes loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Clinical notes file not found at {self.file_path}")
            self.data = pd.DataFrame()

    def get_patient_notes(self, patient_id: str, hadm_id: Optional[str] = None) -> pd.DataFrame:
        """Get all clinical notes for a specific patient, optionally filtered by admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The hospital admission ID to filter by

        Returns:
            DataFrame containing the patient's clinical notes for the specified admission
        """
        if self.data is None:
            self.load_data()

        patient_notes = self.data[self.data['subject_id'] == patient_id].copy()

        if hadm_id:
            patient_notes = patient_notes[patient_notes['hadm_id'] == hadm_id]

        # Sort by charttime
        if not patient_notes.empty and 'charttime' in patient_notes.columns:
            patient_notes = patient_notes.sort_values('charttime')

        return patient_notes

    def search_notes_for_keywords(self, patient_id: str, hadm_id: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search an admission's notes for specific keywords.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            keywords: List of keywords to search for

        Returns:
            List of dictionaries containing matching notes and context
        """
        notes = self.get_patient_notes(patient_id, hadm_id)
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
                    'hadm_id': note.get('hadm_id'),
                    'charttime': note.get('charttime'),
                    'note_type': note.get('note_type'),
                    'matches': note_matches
                })

        return results

    def get_patient_data(self, patient_id: str, hadm_id: Optional[str] = None) -> Dict[str, Any]:
        """Get clinical notes data for a specific patient and admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing the patient's clinical notes data
        """
        notes = self.get_patient_notes(patient_id, hadm_id)
        return {
            'notes': notes.to_dict('records') if not notes.empty else []
        }

    def get_earliest_timestamp(self, patient_id: str, hadm_id: str) -> Optional[pd.Timestamp]:
        """Get the earliest charttime for a given admission."""
        if self.data is None:
            self.load_data()

        admission_notes = self.data[
            (self.data['subject_id'] == patient_id) &
            (self.data['hadm_id'] == hadm_id)
        ]

        if not admission_notes.empty and 'charttime' in admission_notes.columns:
            return admission_notes['charttime'].min()
        
        return None

    def get_all_admissions(self) -> List[Tuple[str, str]]:
        """Get all unique (patient_id, hadm_id) tuples from the notes data.

        Returns:
            A list of (patient_id, hadm_id) tuples.
        """
        if self.data is None:
            self.load_data()

        if 'subject_id' not in self.data.columns or 'hadm_id' not in self.data.columns:
            return []

        # Drop rows where subject_id or hadm_id is missing
        admissions_df = self.data[['subject_id', 'hadm_id']].dropna().drop_duplicates()
        # Convert to list of tuples
        admissions = [tuple(row) for row in admissions_df.itertuples(index=False)]
        return admissions
