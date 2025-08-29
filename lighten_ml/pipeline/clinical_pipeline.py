"""Main pipeline for clinical data processing."""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

from ..data_loaders import (
    LabDataLoader,
    ClinicalNotesLoader
)
from ..evidence_collectors import (
    TroponinAnalyzer,
    ClinicalEvidenceExtractor,
    ECGEvidenceExtractor,
    ImagingEvidenceExtractor,
    AngiographyEvidenceExtractor
)
from ..rule_engines import MIRuleEngine, MIRuleEngineConfig
from ..llm_client import LightenLLMClient
from ..resolvers.onset_date_resolver import OnsetDateResolver

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types like pandas Timestamps."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


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
        
        # Initialize optional LLM client
        llm_cfg = (self.config or {}).get('llm', {})
        api_key = llm_cfg.get('api_key') or os.environ.get('TOGETHER_API_KEY') or os.environ.get('LLM_API_KEY')
        model = llm_cfg.get('model') or os.environ.get('LLM_MODEL')
        base_url = llm_cfg.get('base_url') or os.environ.get('LLM_BASE_URL')
        self.llm_client = LightenLLMClient(
            api_key=api_key, 
            model=model, 
            base_url=base_url,
            rate_limit_tps=llm_cfg.get('rate_limit_tps', 5),
            cache_size=llm_cfg.get('cache_size', 1024)
        ) if (api_key or os.environ.get('TOGETHER_API_KEY') or os.environ.get('LLM_API_KEY')) else None

        # Initialize evidence collectors
        collectors_cfg = self.config.get('evidence_collectors', {})
        troponin_config = collectors_cfg.get('troponin', {})
        max_notes = collectors_cfg.get('max_notes_per_admission')

        self.troponin_analyzer = TroponinAnalyzer(
            self.lab_loader,
            time_window_hours=troponin_config.get('time_window_hours', 72)
        )
        self.clinical_evidence_extractor = ClinicalEvidenceExtractor(self.notes_loader, llm_client=self.llm_client, max_notes=max_notes)
        self.ecg_evidence_extractor = ECGEvidenceExtractor(self.notes_loader, llm_client=self.llm_client, max_notes=max_notes)
        self.imaging_evidence_extractor = ImagingEvidenceExtractor(self.notes_loader, llm_client=self.llm_client, max_notes=max_notes)
        self.angiography_evidence_extractor = AngiographyEvidenceExtractor(self.notes_loader, llm_client=self.llm_client, max_notes=max_notes)
        
        # Initialize rule engine with config
        rule_engine_config = self.config.get('rule_engine', {})
        self.rule_engine = MIRuleEngine(
            MIRuleEngineConfig(**rule_engine_config)
            if rule_engine_config 
            else None
        )

        # Initialize the onset date resolver
        self.onset_date_resolver = OnsetDateResolver()

        # Cache for admission data
        self._admission_cache = {}
    
    def get_available_admissions(self) -> List[Tuple[str, str]]:
        """Get a list of (patient_id, hadm_id) tuples that have both lab and note data."""
        # These methods will need to be implemented in the loaders
        lab_admissions = self.lab_loader.get_all_admissions()
        notes_admissions = self.notes_loader.get_all_admissions()

        # Find the intersection of admissions present in both data sources
        lab_set = set(lab_admissions)
        notes_set = set(notes_admissions)
        common_admissions = sorted(list(lab_set.intersection(notes_set)))

        return common_admissions

    def process_admission(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Process data for a single hospital admission.
        
        Args:
            patient_id: The ID of the patient to process
            hadm_id: The hospital admission ID to process
            
        Returns:
            Dictionary containing processing results
        """
        # Check cache first
        if hadm_id in self._admission_cache:
            return self._admission_cache[hadm_id]
        
        # Initialize result structure
        result = {
            'patient_id': patient_id,
            'hadm_id': hadm_id,
            'timestamp': datetime.utcnow().isoformat(),
            'evidence': {},
            'results': {}
        }
        
        try:
            # 1. Collect all available evidence for the specific admission
            evidence = self._collect_all_evidence(patient_id, hadm_id)
            result['evidence'] = evidence
            
            # 2. Apply rule engine to evaluate MI
            rule_result = self.rule_engine.evaluate(evidence)
            
            # 3. Determine MI Onset Date
            onset_date_result = {}
            if rule_result.passed: # Only resolve onset date if MI is detected
                # Get admission time as a fallback for onset date
                admission_time = self._get_earliest_admission_timestamp(patient_id, hadm_id)

                onset_date_result = self.onset_date_resolver.resolve(result['evidence'], admission_time=admission_time)

            # 4. Format results
            result['results'] = {
                'mi_detected': rule_result.passed,
                'mi_onset_date': onset_date_result.get('onset_date'),
                'mi_onset_date_rationale': onset_date_result.get('rationale'),
                'confidence': rule_result.confidence,
                'details': rule_result.details,
                'timestamp': rule_result.timestamp
            }
            
            # 5. Add summary
            result['summary'] = self._generate_summary(result)
            
            # Cache the result
            self._admission_cache[hadm_id] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing admission {hadm_id} for patient {patient_id}: {str(e)}"
            result['error'] = error_msg
            return result
    
    def _collect_all_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect all available evidence for a given admission."""
        return {
            'troponin': self.troponin_analyzer.collect_evidence(patient_id, hadm_id),
            'clinical': self.clinical_evidence_extractor.collect_evidence(patient_id, hadm_id),
            'ecg': self.ecg_evidence_extractor.collect_evidence(patient_id, hadm_id),
            'imaging': self.imaging_evidence_extractor.collect_evidence(patient_id, hadm_id),
            'angiography': self.angiography_evidence_extractor.collect_evidence(patient_id, hadm_id)
        }

    def _get_earliest_admission_timestamp(self, patient_id: str, hadm_id: str) -> Optional[pd.Timestamp]:
        """Get the earliest timestamp for an admission from labs and notes."""
        earliest_lab_time = self.lab_loader.get_earliest_timestamp(patient_id, hadm_id)
        earliest_note_time = self.notes_loader.get_earliest_timestamp(patient_id, hadm_id)

        if earliest_lab_time and earliest_note_time:
            return min(earliest_lab_time, earliest_note_time)
        
        return earliest_lab_time or earliest_note_time

    def process_admissions(self, admissions: List[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Process multiple hospital admissions.
        
        Args:
            admissions: List of (patient_id, hadm_id) tuples to process
            
        Returns:
            Dictionary mapping admission IDs to their results
        """
        results = {}
        
        for patient_id, hadm_id in admissions:
            result = self.process_admission(patient_id, hadm_id)
            results[hadm_id] = result
            
            # Save individual admission result
            self._save_admission_result(hadm_id, result)
        
        # Save combined results
        self._save_combined_results(results)
        
        return results
    
    def _generate_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a human-readable summary of the results.
        
        Args:
            result: The processing result for an admission
            
        Returns:
            Dictionary containing a summary of the results
        """
        summary = {
            'patient_id': result['patient_id'],
            'hadm_id': result['hadm_id'],
            'mi_detected': result['results']['mi_detected'],
            'mi_onset_date': result['results'].get('mi_onset_date'),
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
        
        # Add imaging findings
        imaging = result['evidence'].get('imaging', {})
        if imaging.get('imaging_findings'):
            mi_related = [f for f in imaging['imaging_findings'] if f.get('mi_related', False)]
            if mi_related:
                summary['key_findings'].append({
                    'category': 'Imaging',
                    'finding': f"{len(mi_related)} MI-related imaging findings",
                    'significance': 'Supports MI diagnosis'
                })
        
        # Add angiography findings
        angiography = result['evidence'].get('angiography', {})
        if angiography.get('angiography_findings'):
            mi_related = [f for f in angiography['angiography_findings'] if f.get('mi_related', False)]
            if mi_related:
                summary['key_findings'].append({
                    'category': 'Angiography',
                    'finding': f"{len(mi_related)} MI-related angiography findings",
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
    
    def _save_admission_result(self, hadm_id: str, result: Dict[str, Any]) -> str:
        """Save a single admission's result to a JSON file.
        
        Args:
            hadm_id: The admission ID
            result: The processing result
            
        Returns:
            Path to the saved file
        """
        filename = os.path.join(self.output_dir, f"admission_{hadm_id}_results.json")
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, cls=CustomJSONEncoder)
        
        return filename
    
    def _save_combined_results(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Save combined results to JSON and CSV files.
        
        Args:
            results: Dictionary mapping admission IDs to their results
            
        Returns:
            Paths to the saved files
        """
        # --- Save Simplified Summary (JSON and CSV) --- 
        summary = {}
        for hadm_id, result in results.items():
            summary[hadm_id] = {
                'patient_id': result.get('patient_id'),
                'mi_detected': result.get('results', {}).get('mi_detected', False),
                'mi_onset_date': result.get('results', {}).get('mi_onset_date'),
                'mi_onset_date_rationale': result.get('results', {}).get('mi_onset_date_rationale'),
                'confidence': result.get('results', {}).get('confidence', 0.0),
                'summary': result.get('summary', {})
            }
        
        json_path = os.path.join(self.output_dir, 'combined_results.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=CustomJSONEncoder)
        
        # --- Save Requirement-Compliant Nested JSON --- 
        requirement_output = {}
        for hadm_id, result in results.items():
            patient_id = result['patient_id']
            if patient_id not in requirement_output:
                requirement_output[patient_id] = {}
            
            mi_result = result['results'].get('mi_detected', 'NO')
            onset_date_result = result['results'].get('mi_onset_date')
            
            # Use the most definitive MI result for a patient
            if mi_result or requirement_output[patient_id].get('Myocardial Infarction') != 'YES':
                requirement_output[patient_id]['Myocardial Infarction'] = 'YES' if mi_result else 'NO'
                
                onset_date_iso = onset_date_result if onset_date_result else None
                if onset_date_iso:
                    # Format to YYYY-MM-DD for the final output, matching the requirement
                    requirement_output[patient_id]['Myocardial Infarction Onset Date'] = pd.to_datetime(onset_date_iso).strftime('%Y-%m-%d')
                else:
                    requirement_output[patient_id]['Myocardial Infarction Onset Date'] = None

        requirement_json_path = os.path.join(self.output_dir, 'requirement_compliant_output.json')
        with open(requirement_json_path, 'w') as f:
            json.dump(requirement_output, f, indent=4, cls=CustomJSONEncoder)

        # --- Save to CSV --- 
        csv_path = os.path.join(self.output_dir, 'combined_results.csv')
        rows = []
        for hadm_id, data in summary.items():
            row = {
                'hadm_id': hadm_id,
                'patient_id': data['patient_id'],
                'mi_detected': data['mi_detected'],
                'mi_onset_date': data['mi_onset_date'],
                'mi_onset_date_rationale': data['mi_onset_date_rationale'],
                'confidence': data['confidence']
            }
            
            findings = data.get('summary', {}).get('key_findings', [])
            row['key_findings'] = '; '.join(
                f"{f.get('category', '')}: {f.get('finding', '')}" 
                for f in findings
            )
            
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        
        return json_path, csv_path, requirement_json_path
    
    def clear_cache(self) -> None:
        """Clear the admission cache."""
        self._admission_cache = {}
