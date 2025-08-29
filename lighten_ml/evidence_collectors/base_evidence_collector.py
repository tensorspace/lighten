"""Base class for evidence collectors."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class BaseEvidenceCollector(ABC):
    """Abstract base class for all evidence collectors."""
    
    def __init__(self, lab_data_loader: Any = None, notes_loader: Any = None):
        """Initialize the evidence collector with optional data loaders.
        
        Args:
            lab_data_loader: Instance of LabDataLoader for accessing lab data
            notes_loader: Instance of ClinicalNotesLoader for accessing clinical notes
        """
        self.lab_data_loader = lab_data_loader
        self.notes_loader = notes_loader
    
    @abstractmethod
    def collect_evidence(self, patient_id: str) -> Dict[str, Any]:
        """Collect evidence for a specific patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing collected evidence
        """
        pass
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.
        
        Returns:
            ISO formatted current timestamp
        """
        return datetime.utcnow().isoformat()
    
    def _get_evidence_base(self) -> Dict[str, Any]:
        """Get base evidence structure.
        
        Returns:
            Dictionary with base evidence structure
        """
        return {
            'timestamp': self._get_timestamp(),
            'sources': [],
            'confidence': None,
            'metadata': {}
        }
