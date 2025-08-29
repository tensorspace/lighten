"""Base class for evidence collectors."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class BaseEvidenceCollector(ABC):
    """Abstract base class for all evidence collectors."""
    
    def __init__(self, 
                 lab_data_loader: Optional[Any] = None, 
                 notes_data_loader: Optional[Any] = None,
                 llm_client: Optional[Any] = None,
                 max_notes: Optional[int] = None):
        """Initialize the base evidence collector.
        
        Args:
            lab_data_loader: Instance of LabDataLoader
            notes_data_loader: Instance of ClinicalNotesLoader
            llm_client: Instance of LightenLLMClient
            max_notes: Maximum number of notes to process per admission
        """
        self.lab_data_loader = lab_data_loader
        self.notes_data_loader = notes_data_loader
        self.llm_client = llm_client
        self.max_notes = max_notes
    
    @abstractmethod
    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Abstract method to collect evidence for a specific patient and admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing collected evidence
        """
        raise NotImplementedError("Subclasses must implement collect_evidence")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.
        
        Returns:
            ISO formatted current timestamp
        """
        return datetime.utcnow().isoformat()
    
    def _get_evidence_base(self) -> Dict[str, Any]:
        """Get a base dictionary for evidence results."""
        return {
            'timestamp': self._get_timestamp(),
            'sources': [],
            'confidence': None,
            'metadata': {}
        }
