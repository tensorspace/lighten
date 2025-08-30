"""Lab data loading and processing."""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from .base_loader import BaseDataLoader
import logging

logger = logging.getLogger(__name__)

class LabDataLoader(BaseDataLoader):
    """Load and process laboratory test results."""
    
    def __init__(self, lab_events_path: str, d_labitems_path: str):
        """Initialize the lab data loader.
        
        Args:
            lab_events_path: Path to the lab events CSV file
            d_labitems_path: Path to the lab items dictionary CSV file
        """
        self.lab_events_path = lab_events_path
        self.d_labitems_path = d_labitems_path
        self.lab_items: Optional[pd.DataFrame] = None
        super().__init__(lab_events_path)
    
    def load_data(self) -> None:
        """Load and preprocess lab data."""
        logger.info(f"Loading lab events from {self.lab_events_path}...")
        try:
            # Load lab items mapping
            self.lab_items = pd.read_csv(self.d_labitems_path)
        except FileNotFoundError:
            logger.error(f"Lab items file not found at {self.d_labitems_path}")
            self.lab_items = pd.DataFrame()

        # Define dtypes for ID columns
        dtypes = {'subject_id': str, 'hadm_id': str}

        # Load lab events in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(self.lab_events_path, chunksize=10000, dtype=dtypes):
            # Merge with lab items to get test names
            chunk = pd.merge(
                chunk,
                self.lab_items[['itemid', 'label', 'fluid', 'category']],
                on='itemid',
                how='left'
            )
            chunks.append(chunk)

        self.data = pd.concat(chunks) if chunks else pd.DataFrame()

        # Convert charttime to datetime
        if 'charttime' in self.data.columns:
            self.data['charttime'] = pd.to_datetime(self.data['charttime'])

        logger.info("Lab data loaded successfully.")

    def get_patient_data(self, patient_id: str, hadm_id: Optional[str] = None) -> Dict[str, Any]:
        """Get lab data for a specific patient and admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The hospital admission ID to filter by

        Returns:
            Dictionary containing the patient's lab data
        """
        if self.data is None:
            self.load_data()

        patient_data = self.data[self.data['subject_id'] == patient_id].copy()

        if hadm_id:
            patient_data = patient_data[patient_data['hadm_id'] == hadm_id]

        # Sort by charttime
        if 'charttime' in patient_data.columns:
            patient_data = patient_data.sort_values('charttime')

        return {
            'lab_events': patient_data.to_dict('records')
        }

    def get_troponin_tests(self, patient_id: str, hadm_id: str) -> pd.DataFrame:
        """Get troponin test results for a specific admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            DataFrame containing troponin test results
        """
        if self.data is None:
            self.load_data()

        logger.info(f"[{hadm_id}] === TROPONIN DATA SEARCH DEBUG ===")
        logger.info(f"[{hadm_id}] Searching for patient_id='{patient_id}', hadm_id='{hadm_id}'")
        logger.info(f"[{hadm_id}] Total lab records in dataset: {len(self.data)}")
        
        # Check what data exists for this patient/admission combination
        patient_admission_data = self.data[
            (self.data['subject_id'] == patient_id) &
            (self.data['hadm_id'] == hadm_id)
        ]
        logger.info(f"[{hadm_id}] Lab records for this patient/admission: {len(patient_admission_data)}")
        
        if not patient_admission_data.empty:
            unique_labels = patient_admission_data['label'].unique()
            logger.info(f"[{hadm_id}] Available lab test types for this admission: {list(unique_labels)[:10]}...")  # Show first 10
            
            # Check specifically for troponin-related tests
            troponin_labels = [label for label in unique_labels if 'troponin' in str(label).lower()]
            logger.info(f"[{hadm_id}] Troponin-related labels found: {troponin_labels}")
        else:
            logger.warning(f"[{hadm_id}] No lab data found for patient_id='{patient_id}', hadm_id='{hadm_id}'")
            logger.info(f"[{hadm_id}] Sample patient_ids in dataset: {list(self.data['subject_id'].unique())[:5]}")
            logger.info(f"[{hadm_id}] Sample hadm_ids in dataset: {list(self.data['hadm_id'].unique())[:5]}")

        # Get troponin tests for the specified admission (case insensitive match)
        troponin_tests = self.data[
            (self.data['subject_id'] == patient_id) &
            (self.data['hadm_id'] == hadm_id) &
            (self.data['label'].str.contains('troponin', case=False, na=False))
        ].copy()
        
        logger.info(f"[{hadm_id}] Final troponin test records found: {len(troponin_tests)}")
        
        if not troponin_tests.empty:
            values = troponin_tests['valuenum'].dropna()
            logger.info(f"[{hadm_id}] Troponin values found: {list(values)}")
            logger.info(f"[{hadm_id}] Troponin value range: min={values.min():.6f}, max={values.max():.6f}")
            logger.info(f"[{hadm_id}] Diagnostic threshold: 0.014 ng/mL")
            above_threshold = values[values > 0.014]
            logger.info(f"[{hadm_id}] Values above threshold: {len(above_threshold)} out of {len(values)}")

        # Sort by charttime
        if not troponin_tests.empty and 'charttime' in troponin_tests.columns:
            troponin_tests = troponin_tests.sort_values('charttime')

        return troponin_tests

    def get_lab_tests_by_name(self, patient_id: str, hadm_id: str, test_name: str) -> pd.DataFrame:
        """Get specific lab test results for an admission by test name.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            test_name: Name of the test to retrieve (case insensitive)

        Returns:
            DataFrame containing the requested test results
        """
        if self.data is None:
            self.load_data()

        tests = self.data[
            (self.data['subject_id'] == patient_id) &
            (self.data['hadm_id'] == hadm_id) &
            (self.data['label'].str.contains(test_name, case=False, na=False))
        ].copy()

        if not tests.empty and 'charttime' in tests.columns:
            tests = tests.sort_values('charttime')

        return tests

    def get_earliest_timestamp(self, patient_id: str, hadm_id: str) -> Optional[pd.Timestamp]:
        """Get the earliest charttime for a given admission."""
        if self.data is None:
            self.load_data()

        admission_data = self.data[
            (self.data['subject_id'] == patient_id) &
            (self.data['hadm_id'] == hadm_id)
        ]

        if not admission_data.empty and 'charttime' in admission_data.columns:
            return admission_data['charttime'].min()
        
        return None

    def get_all_admissions(self) -> List[Tuple[str, str]]:
        """Get all unique (patient_id, hadm_id) tuples from the lab data.

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
