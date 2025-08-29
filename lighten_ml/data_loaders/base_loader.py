"""Base class for data loaders."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    
    def __init__(self, file_path: str, **kwargs):
        """Initialize the data loader with a file path.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for specific loaders
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def load_data(self) -> None:
        """Load and preprocess the data."""
        pass
    
    @abstractmethod
    def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get data for a specific patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing the patient's data
        """
        pass
    
    def get_all_patient_ids(self) -> List[str]:
        """Get a list of all patient IDs in the dataset.
        
        Returns:
            List of patient IDs
        """
        if self.data is None:
            self.load_data()
        return self.data['subject_id'].unique().tolist()
