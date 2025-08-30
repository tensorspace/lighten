#!/usr/bin/env python3
"""
Command-line interface for running the Lighten ML pipeline.
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from lighten_ml.pipeline import ClinicalPipeline


def main():
    """Main entry point for the script."""
    load_dotenv()  # Load environment variables from .env file

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Process clinical data for MI detection"
    )
    parser.add_argument(
        "--lab-events",
        type=str,
        default="labevents.csv",
        help="Path to lab events CSV file",
    )
    parser.add_argument(
        "--lab-items",
        type=str,
        default="d_labitems.csv",
        help="Path to lab items dictionary CSV file",
    )
    parser.add_argument(
        "--clinical-notes",
        type=str,
        default="discharge_notes_demo.csv",
        help="Path to clinical notes CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Process a specific patient ID (subject_id)",
    )
    parser.add_argument(
        "--hadm-id",
        type=str,
        default=None,
        help="Process a specific hospital admission ID (hadm_id)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process a random sample of N admissions",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all available admissions"
    )
    # LLM options (API key is read from .env file)
    parser.add_argument(
        "--llm-model", type=str, default=None, help="LLM model name (overrides default)"
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM base URL (default: Together chat completions endpoint)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the LLM cache before processing",
    )

    args = parser.parse_args()

    # Check if input files exist
    for path in [args.lab_events, args.lab_items, args.clinical_notes]:
        if not os.path.exists(path):
            print(f"Error: Input file not found: {path}")
            sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Clear LLM cache if requested
    if args.clear_cache:
        cache_file = os.path.join(args.output_dir, "llm_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Cleared LLM cache: {cache_file}")
        else:
            print("No LLM cache file found to clear.")

    try:
        # Initialize the pipeline
        print("Initializing pipeline...")
        pipeline = ClinicalPipeline(
            lab_events_path=args.lab_events,
            lab_items_path=args.lab_items,
            clinical_notes_path=args.clinical_notes,
            output_dir=args.output_dir,
            config={
                "llm": {
                    "model": args.llm_model,
                    "base_url": args.llm_base_url,
                }
            },
        )

        # Get patient admissions to process
        all_admissions = pipeline.get_available_admissions()

        if not all_admissions:
            print("No admissions found with both lab data and clinical notes.")
            sys.exit(1)

        if args.hadm_id:
            admissions_to_process = [
                (pid, hid) for pid, hid in all_admissions if hid == args.hadm_id
            ]
            if not admissions_to_process:
                print(
                    f"Error: Admission ID {args.hadm_id} not found or is missing required data."
                )
                sys.exit(1)
        elif args.patient_id:
            admissions_to_process = [
                (pid, hid) for pid, hid in all_admissions if pid == args.patient_id
            ]
            if not admissions_to_process:
                print(
                    f"Error: Patient ID {args.patient_id} not found or is missing required data."
                )
                sys.exit(1)
        elif args.sample:
            if args.sample > len(all_admissions):
                print(
                    f"Warning: Sample size {args.sample} is larger than available admissions ({len(all_admissions)})."
                )
                args.sample = len(all_admissions)
            indices = random.sample(range(len(all_admissions)), args.sample)
            admissions_to_process = [all_admissions[i] for i in indices]
        elif args.all:
            admissions_to_process = all_admissions
        else:
            # Default to processing the first admission if no other option is specified
            admissions_to_process = all_admissions[:1]

        print(f"Processing {len(admissions_to_process)} admission(s)...")
        pipeline.process_admissions(admissions_to_process)

        print("Processing complete.")
        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
