#!/usr/bin/env python3
"""
Command-line interface for running the Lighten ML pipeline.
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from lighten_ml.pipeline import ClinicalPipeline

def main():
    parser = argparse.ArgumentParser(description='Process clinical data for MI detection')
    parser.add_argument('--lab-events', type=str, default='labevents.csv',
                        help='Path to lab events CSV file')
    parser.add_argument('--lab-items', type=str, default='d_labitems.csv',
                        help='Path to lab items dictionary CSV file')
    parser.add_argument('--clinical-notes', type=str, default='discharge_notes_demo.csv',
                        help='Path to clinical notes CSV file')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--patient', type=str, default=None,
                        help='Process a specific patient ID')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process a random sample of N patients')
    parser.add_argument('--all', action='store_true',
                        help='Process all available patients')
    # LLM options
    parser.add_argument('--llm-api-key', type=str, default=None,
                        help='LLM API key (or set TOGETHER_API_KEY/LLM_API_KEY env var)')
    parser.add_argument('--llm-model', type=str, default=None,
                        help='LLM model name (overrides default)')
    parser.add_argument('--llm-base-url', type=str, default=None,
                        help='LLM base URL (default: Together chat completions endpoint)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for path in [args.lab_events, args.lab_items, args.clinical_notes]:
        if not os.path.exists(path):
            print(f"Error: Input file not found: {path}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize the pipeline
        print("Initializing pipeline...")
        pipeline = ClinicalPipeline(
            lab_events_path=args.lab_events,
            lab_items_path=args.lab_items,
            clinical_notes_path=args.clinical_notes,
            output_dir=args.output_dir,
            config={
                'llm': {
                    'api_key': args.llm_api_key,
                    'model': args.llm_model,
                    'base_url': args.llm_base_url,
                }
            }
        )
        
        # Get patient IDs to process
        all_patient_ids = pipeline.get_available_patient_ids()
        
        if not all_patient_ids:
            print("No patients found with both lab data and clinical notes.")
            sys.exit(1)
        
        if args.patient:
            # Process specific patient
            if args.patient not in all_patient_ids:
                print(f"Patient {args.patient} not found in the dataset.")
                sys.exit(1)
            patient_ids = [args.patient]
        elif args.sample:
            # Process random sample
            import random
            sample_size = min(args.sample, len(all_patient_ids))
            patient_ids = random.sample(all_patient_ids, sample_size)
        elif args.all:
            # Process all patients
            patient_ids = all_patient_ids
        else:
            # Default: process first patient
            patient_ids = all_patient_ids[:1]
        
        print(f"Processing {len(patient_ids)} patient(s)...")
        
        # Process patients
        results = {}
        for patient_id in tqdm(patient_ids, desc="Processing patients"):
            try:
                result = pipeline.process_patient(patient_id)
                results[patient_id] = result
                
                # Print summary for each patient
                summary = result.get('summary', {})
                print(f"\nPatient {patient_id}:")
                print(f"  MI Detected: {summary.get('mi_detected', 'N/A')}")
                print(f"  Confidence: {summary.get('confidence', 'N/A'):.2f}")
                
                # Print key findings
                if 'key_findings' in summary:
                    print("  Key Findings:")
                    for finding in summary['key_findings']:
                        print(f"    - {finding.get('category')}: {finding.get('finding')}")
            
            except Exception as e:
                print(f"Error processing patient {patient_id}: {str(e)}")
        
        # Save combined results
        if results:
            pipeline._save_combined_results(results)
            print(f"\nResults saved to {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
