"""Data loader for admissions data."""
from typing import List, Tuple, Optional
import pandas as pd
from .base_loader import BaseDataLoader

class AdmissionsLoader(BaseDataLoader):
    """Load and process admissions data."""

    def __init__(self, admissions_path: str):
        super().__init__(admissions_path)
        self.admissions_columns = ['subject_id', 'hadm_id', 'admittime']

    def load_data(self) -> None:
        self.data = pd.read_csv(
            self.file_path,
            usecols=self.admissions_columns,
            parse_dates=['admittime'],
            dtype={'subject_id': str, 'hadm_id': str}
        )

    def get_admission_time(self, patient_id: str, hadm_id: str) -> Optional[pd.Timestamp]:
        """Get the admission time for a specific admission."""
        if self.data is None:
            self.load_data()

        admission = self.data[
            (self.data['subject_id'] == patient_id) & (self.data['hadm_id'] == hadm_id)
        ]

        if not admission.empty:
            return admission.iloc[0]['admittime']
        return None

    def get_all_admissions(self) -> List[Tuple[str, str]]:
        """Get all unique (patient_id, hadm_id) tuples."""
        if self.data is None:
            self.load_data()

        admissions_df = self.data[['subject_id', 'hadm_id']].dropna().drop_duplicates()
        return [tuple(row) for row in admissions_df.itertuples(index=False)]

    def get_patient_data(self, patient_id: str, hadm_id: Optional[str] = None) -> pd.DataFrame:
        """Get admissions data for a patient."""
        if self.data is None:
            self.load_data()

        patient_admissions = self.data[self.data['subject_id'] == patient_id]
        if hadm_id:
            patient_admissions = patient_admissions[patient_admissions['hadm_id'] == hadm_id]

        return patient_admissions
