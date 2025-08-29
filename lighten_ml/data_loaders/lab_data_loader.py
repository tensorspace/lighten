"""Lab data loading and processing."""
from typing import Dict, List, Optional, Any
import pandas as pd
from .base_loader import BaseDataLoader

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
        # Load lab items mapping
        self.lab_items = pd.read_csv(self.d_labitems_path)
        
        # Load lab events in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(self.lab_events_path, chunksize=10000):
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
    
    def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get lab data for a specific patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing the patient's lab data
        """
        if self.data is None:
            self.load_data()
            
        patient_data = self.data[self.data['subject_id'] == patient_id].copy()
        
        # Sort by charttime
        if 'charttime' in patient_data.columns:
            patient_data = patient_data.sort_values('charttime')
            
        return {
            'lab_events': patient_data.to_dict('records')
        }
    
    def get_troponin_tests(self, patient_id: str) -> pd.DataFrame:
        """Get troponin test results for a specific patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            DataFrame containing troponin test results
        """
        if self.data is None:
            self.load_data()
            
        # Get troponin tests (case insensitive match)
        troponin_tests = self.data[
            (self.data['subject_id'] == patient_id) & 
            (self.data['label'].str.contains('troponin', case=False, na=False))
        ].copy()
        
        # Sort by charttime
        if not troponin_tests.empty and 'charttime' in troponin_tests.columns:
            troponin_tests = troponin_tests.sort_values('charttime')
            
        return troponin_tests
    
    def get_lab_tests_by_name(self, patient_id: str, test_name: str) -> pd.DataFrame:
        """Get specific lab test results for a patient by test name.
        
        Args:
            patient_id: The ID of the patient
            test_name: Name of the test to retrieve (case insensitive)
            
        Returns:
            DataFrame containing the requested test results
        """
        if self.data is None:
            self.load_data()
            
        tests = self.data[
            (self.data['subject_id'] == patient_id) & 
            (self.data['label'].str.contains(test_name, case=False, na=False))
        ].copy()
        
        if not tests.empty and 'charttime' in tests.columns:
            tests = tests.sort_values('charttime')
            
        return tests
