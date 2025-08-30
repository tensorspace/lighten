#!/usr/bin/env python3
"""Test script to verify comprehensive lab test identification system."""

import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lighten_ml.data_loaders.lab_data_loader import LabDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_lab_identification():
    """Test the comprehensive lab test identification system."""
    
    # Initialize the lab data loader
    lab_events_path = "labevents.csv"  # Adjust path as needed
    d_labitems_path = "d_labitems.csv"
    
    logger.info("Initializing LabDataLoader...")
    lab_loader = LabDataLoader(lab_events_path, d_labitems_path)
    
    # Test 1: Check available lab test types
    logger.info("=== TEST 1: Available Lab Test Types ===")
    available_types = lab_loader.get_available_lab_test_types()
    logger.info(f"Available lab test types ({len(available_types)}): {available_types}")
    
    # Test 2: Check itemid mappings
    logger.info("=== TEST 2: Itemid Mappings ===")
    for test_type, itemids in lab_loader.LAB_TEST_ITEMIDS.items():
        logger.info(f"{test_type}: {itemids}")
    
    # Test 3: Try to load data and check structure
    logger.info("=== TEST 3: Data Loading Test ===")
    try:
        lab_loader.load_data()
        if lab_loader.data is not None and not lab_loader.data.empty:
            logger.info(f"Lab data loaded successfully: {len(lab_loader.data)} records")
            logger.info(f"Columns: {list(lab_loader.data.columns)}")
            
            # Check unique itemids in data
            unique_itemids = lab_loader.data['itemid'].unique()
            logger.info(f"Total unique itemids in dataset: {len(unique_itemids)}")
            
            # Check which of our mapped itemids are actually present
            all_mapped_itemids = set()
            for itemids in lab_loader.LAB_TEST_ITEMIDS.values():
                all_mapped_itemids.update(itemids)
            
            present_itemids = set(unique_itemids) & all_mapped_itemids
            logger.info(f"Mapped itemids present in data: {len(present_itemids)} out of {len(all_mapped_itemids)}")
            logger.info(f"Present itemids: {sorted(present_itemids)}")
            
            # Test with a sample patient if data exists
            if len(lab_loader.data) > 0:
                sample_patient = lab_loader.data.iloc[0]['subject_id']
                sample_hadm = lab_loader.data.iloc[0]['hadm_id']
                logger.info(f"Testing with sample patient: {sample_patient}, admission: {sample_hadm}")
                
                # Test troponin specifically
                troponin_tests = lab_loader.get_troponin_tests(sample_patient, sample_hadm)
                logger.info(f"Troponin tests found: {len(troponin_tests)}")
                
                # Test other lab tests
                for test_type in ['glucose', 'creatinine', 'hemoglobin']:
                    if test_type in lab_loader.LAB_TEST_ITEMIDS:
                        results = lab_loader.get_lab_tests_by_type(sample_patient, sample_hadm, test_type)
                        logger.info(f"{test_type.title()} tests found: {len(results)}")
                
                # Test comprehensive summary
                summary = lab_loader.get_comprehensive_lab_summary(sample_patient, sample_hadm)
                logger.info(f"Comprehensive lab summary generated for {sample_patient}")
                logger.info(f"Total tests across all categories: {sum(summary['test_counts'].values())}")
        else:
            logger.warning("No lab data loaded or data is empty")
            
    except FileNotFoundError as e:
        logger.warning(f"Lab data files not found: {e}")
        logger.info("This is expected if running without actual data files")
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
    
    logger.info("=== Lab Test Identification System Test Complete ===")

if __name__ == "__main__":
    test_lab_identification()
