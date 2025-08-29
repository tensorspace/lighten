"""Main pipeline for clinical data processing."""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

from ..data_loaders import LabDataLoader, ClinicalNotesLoader
from ..evidence_collectors import (
    TroponinAnalyzer,
    ClinicalEvidenceExtractor,
    ECGEvidenceExtractor
)
from ..rule_engines import MIRuleEngine, MIRuleEngineConfig

class ClinicalPipeline:
    """Main pipeline for processing clinical data to detect Myocardial Infarction."""
    
    def __init__(self, 
                 lab_events_path: str, 
                 lab_items_path: str, 
                 clinical_notes_path: str,
                 output_dir: str = 'output',
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the clinical pipeline.
        
        Args:
            lab_events_path: Path to the lab events CSV file
            lab_items_path: Path to the lab items dictionary CSV file
            clinical_notes_path: Path to the clinical notes CSV file
            output_dir: Directory to save output files
            config: Optional configuration dictionary
        """
        self.lab_events_path = lab_events_path
        self.lab_items_path = lab_items_path
        self.clinical_notes_path = clinical_notes_path
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data loaders
        self.lab_loader = LabDataLoader(lab_events_path, lab_items_path)
        self.notes_loader = ClinicalNotesLoader(clinical_notes_path)
        
        # Initialize evidence collectors
        self.troponin_analyzer = TroponinAnalyzer(self.lab_loader)
        self.clinical_evidence_extractor = ClinicalEvidenceExtractor(self.notes_loader)
        self.ecg_evidence_extractor = ECGEvidenceExtractor(self.notes_loader)
        
        # Initialize rule engine with config
        rule_engine_config = self.config.get('rule_engine', {})
        self.rule_engine = MIRuleEngine(
            MIRuleEngineConfig(**rule_engine_config)
            if rule_engine_config 
            else None
        )
        
        # Cache for patient data
        self._patient_cache = {}
    
    def process_patient(self, patient_id: str) -> Dict[str, Any]:
        """Process data for a single patient.
        
        Args:
            patient_id: The ID of the patient to process
            
        Returns:
            Dictionary containing processing results
        """
        # Check cache first
        if patient_id in self._patient_cache:
            return self._patient_cache[patient_id]
        
        # Initialize result structure
        result = {
            'patient_id': patient_id,
            'timestamp': datetime.utcnow().isoformat(),
            'evidence': {},
            'results': {}
        }
        
        try:
            # 1. Collect troponin evidence
            troponin_evidence = self.troponin_analyzer.collect_evidence(patient_id)
            result['evidence']['troponin'] = troponin_evidence
            
            # 2. Collect clinical evidence from notes
            clinical_evidence = self.clinical_evidence_extractor.collect_evidence(patient_id)
            result['evidence']['clinical'] = clinical_evidence
            
            # 3. Collect ECG evidence from notes
            ecg_evidence = self.ecg_evidence_extractor.collect_evidence(patient_id)
            result['evidence']['ecg'] = ecg_evidence
            
            # 4. Apply rule engine to evaluate MI
            rule_result = self.rule_engine.evaluate({
                'troponin': troponin_evidence,
                'symptoms': clinical_evidence.get('symptoms', []),
                'ecg': ecg_evidence,
                'imaging': {},  # Placeholder for future imaging integration
                'angiography': {}  # Placeholder for future angiography integration
            })
            
            # 5. Format results
            result['results'] = {
                'mi_detected': rule_result.passed,
                'confidence': rule_result.confidence,
                'details': rule_result.details,
                'timestamp': rule_result.timestamp
            }
            
            # 6. Add summary
            result['summary'] = self._generate_summary(result)
            
            # Cache the result
            self._patient_cache[patient_id] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing patient {patient_id}: {str(e)}"
            result['error'] = error_msg
            return result
    
    def process_patients(self, patient_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Process multiple patients.
        
        Args:
            patient_ids: List of patient IDs to process
            
        Returns:
            Dictionary mapping patient IDs to their results
        """
        results = {}
        
        for patient_id in patient_ids:
            result = self.process_patient(patient_id)
            results[patient_id] = result
            
            # Save individual patient result
            self._save_patient_result(patient_id, result)
        
        # Save combined results
        self._save_combined_results(results)
        
        return results
    
    def _generate_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a human-readable summary of the results.
        
        Args:
            result: The processing result for a patient
            
        Returns:
            Dictionary containing a summary of the results
        """
        summary = {
            'patient_id': result['patient_id'],
            'mi_detected': result['results']['mi_detected'],
            'confidence': result['results']['confidence'],
            'key_findings': []
        }
        
        # Add troponin findings
        troponin = result['evidence'].get('troponin', {})
        if troponin.get('troponin_available', False):
            max_trop = troponin.get('max_troponin', 0)
            threshold = self.rule_engine.config.troponin_threshold
            
            if max_trop > threshold:
                summary['key_findings'].append({
                    'category': 'Troponin',
                    'finding': f'Elevated troponin: {max_trop:.3f} ng/mL (threshold: {threshold} ng/mL)',
                    'significance': 'Supports MI diagnosis'
                })
        
        # Add clinical findings
        clinical = result['evidence'].get('clinical', {})
        if clinical.get('symptoms'):
            symptoms = [s['symptom'] for s in clinical['symptoms']]
            summary['key_findings'].append({
                'category': 'Symptoms',
                'finding': ', '.join(symptoms),
                'significance': 'Consistent with cardiac ischemia' if symptoms else 'No typical symptoms'
            })
        
        # Add ECG findings
        ecg = result['evidence'].get('ecg', {})
        if ecg.get('ecg_findings'):
            mi_related = [f for f in ecg['ecg_findings'] if f.get('mi_related', False)]
            if mi_related:
                summary['key_findings'].append({
                    'category': 'ECG',
                    'finding': f"{len(mi_related)} MI-related ECG findings",
                    'significance': 'Supports MI diagnosis'
                })
        
        # Add rule engine details
        details = result['results'].get('details', {})
        criteria_a = details.get('criteria_A', {})
        criteria_b = details.get('criteria_B', {})
        
        summary['criteria'] = {
            'biomarker_criteria_met': criteria_a.get('met', False),
            'ischemia_criteria_met': criteria_b.get('met', False),
            'required_both_criteria': self.rule_engine.config.require_both_criteria
        }
        
        return summary
    
    def _save_patient_result(self, patient_id: str, result: Dict[str, Any]) -> str:
        """Save a single patient's result to a JSON file.
        
        Args:
            patient_id: The patient ID
            result: The processing result
            
        Returns:
            Path to the saved file
        """
        filename = os.path.join(self.output_dir, f"patient_{patient_id}_results.json")
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        return filename
    
    def _save_combined_results(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Save combined results to a JSON file.
        
        Args:
            results: Dictionary mapping patient IDs to their results
            
        Returns:
            Path to the saved file
        """
        # Create a simplified summary for all patients
        summary = {}
        
        for patient_id, result in results.items():
            summary[patient_id] = {
                'mi_detected': result.get('results', {}).get('mi_detected', False),
                'confidence': result.get('results', {}).get('confidence', 0.0),
                'summary': result.get('summary', {})
            }
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, 'combined_results.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'combined_results.csv')
        rows = []
        
        for patient_id, data in summary.items():
            row = {
                'patient_id': patient_id,
                'mi_detected': data['mi_detected'],
                'confidence': data['confidence']
            }
            
            # Add key findings as a string
            findings = data.get('summary', {}).get('key_findings', [])
            row['key_findings'] = '; '.join(
                f"{f.get('category', '')}: {f.get('finding', '')}" 
                for f in findings
            )
            
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        
        return json_path, csv_path
    
    def clear_cache(self) -> None:
        """Clear the patient cache."""
        self._patient_cache = {}
    
    def get_available_patient_ids(self) -> List[str]:
        """Get a list of available patient IDs from the data loaders.
        
        Returns:
            List of patient IDs
        """
        # Get patient IDs with lab data
        lab_patients = set()
        if hasattr(self.lab_loader, 'data') and 'subject_id' in self.lab_loader.data.columns:
            lab_patients = set(self.lab_loader.data['subject_id'].unique())
        
        # Get patient IDs with clinical notes
        notes_patients = set()
        if hasattr(self.notes_loader, 'data') and 'subject_id' in self.notes_loader.data.columns:
            notes_patients = set(self.notes_loader.data['subject_id'].unique())
        
        # Return intersection of patients with both lab data and notes
        return sorted(list(lab_patients.intersection(notes_patients)))
