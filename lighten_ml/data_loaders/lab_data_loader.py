"""Lab data loading and processing."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class LabDataLoader(BaseDataLoader):
    """Load and process laboratory test results."""

    # Simple itemid-to-label mapping system using d_labitems.csv
    _itemid_to_label_map: Optional[Dict[int, Dict[str, str]]] = None

    def __init__(self, lab_events_path: str, d_labitems_path: str):
        """Initialize the lab data loader.

        Args:
            lab_events_path: Path to the lab events CSV file
            d_labitems_path: Path to the lab items dictionary CSV file
        """
        self.lab_events_path = lab_events_path
        self.d_labitems_path = d_labitems_path
        self.lab_items: Optional[pd.DataFrame] = None
        self._itemid_to_label_map = None
        super().__init__(lab_events_path)

    def load_data(self) -> None:
        """Load and preprocess lab data."""
        logger.info("=== Starting Lab Data Loading Process ===")
        logger.info(f"Lab events file: {self.lab_events_path}")
        logger.info(f"Lab items file: {self.d_labitems_path}")

        # Load lab items mapping
        logger.info("Step 1: Loading lab items mapping...")
        try:
            self.lab_items = pd.read_csv(self.d_labitems_path)
            logger.info(f"✅ Lab items loaded: {len(self.lab_items)} items")
            logger.info(f"Lab items columns: {list(self.lab_items.columns)}")

            # Log some statistics about lab items
            if not self.lab_items.empty:
                unique_categories = (
                    self.lab_items["category"].nunique()
                    if "category" in self.lab_items.columns
                    else 0
                )
                unique_fluids = (
                    self.lab_items["fluid"].nunique()
                    if "fluid" in self.lab_items.columns
                    else 0
                )
                logger.info(
                    f"Lab items stats: {unique_categories} categories, {unique_fluids} fluid types"
                )
        except FileNotFoundError:
            logger.error(f"❌ Lab items file not found at {self.d_labitems_path}")
            self.lab_items = pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error loading lab items: {e}")
            self.lab_items = pd.DataFrame()

        # Define dtypes for ID columns
        dtypes = {"subject_id": str, "hadm_id": str}
        logger.info("Step 2: Loading lab events...")
        logger.info(
            "Using chunked loading (10,000 rows per chunk) for memory efficiency"
        )

        # Load lab events in chunks to handle large files
        chunks = []
        total_rows_processed = 0
        chunk_count = 0

        try:
            for chunk in pd.read_csv(
                self.lab_events_path, chunksize=10000, dtype=dtypes
            ):
                chunk_count += 1
                chunk_size = len(chunk)
                total_rows_processed += chunk_size

                logger.info(
                    f"Processing chunk {chunk_count}: {chunk_size} rows (total: {total_rows_processed})"
                )

                # Log chunk statistics for first chunk
                if chunk_count == 1:
                    logger.info(f"Lab events columns: {list(chunk.columns)}")
                    unique_patients = (
                        chunk["subject_id"].nunique()
                        if "subject_id" in chunk.columns
                        else 0
                    )
                    unique_admissions = (
                        chunk["hadm_id"].nunique() if "hadm_id" in chunk.columns else 0
                    )
                    unique_itemids = (
                        chunk["itemid"].nunique() if "itemid" in chunk.columns else 0
                    )
                    logger.info(
                        f"First chunk stats: {unique_patients} patients, {unique_admissions} admissions, {unique_itemids} unique itemids"
                    )

                # Merge with lab items to get test names
                if not self.lab_items.empty:
                    before_merge = len(chunk)
                    chunk = pd.merge(
                        chunk,
                        self.lab_items[["itemid", "label", "fluid", "category"]],
                        on="itemid",
                        how="left",
                    )
                    after_merge = len(chunk)

                    # Check merge success
                    merged_with_labels = chunk["label"].notna().sum()
                    logger.info(
                        f"Chunk {chunk_count} merge: {merged_with_labels}/{after_merge} rows got labels ({merged_with_labels/after_merge*100:.1f}%)"
                    )
                else:
                    logger.warning(
                        f"Chunk {chunk_count}: No lab items available for merging"
                    )

                chunks.append(chunk)

                # Log progress every 10 chunks
                if chunk_count % 10 == 0:
                    logger.info(
                        f"Progress: {chunk_count} chunks processed, {total_rows_processed} total rows"
                    )

        except FileNotFoundError:
            logger.error(f"❌ Lab events file not found: {self.lab_events_path}")
            self.data = pd.DataFrame()
            return
        except Exception as e:
            logger.error(f"❌ Error loading lab events: {e}")
            self.data = pd.DataFrame()
            return

        # Concatenate all chunks
        logger.info(f"Step 3: Concatenating {len(chunks)} chunks...")
        self.data = pd.concat(chunks) if chunks else pd.DataFrame()

        if not self.data.empty:
            logger.info(f"✅ Lab events concatenated: {len(self.data)} total records")

            # Log final dataset statistics
            unique_patients = (
                self.data["subject_id"].nunique()
                if "subject_id" in self.data.columns
                else 0
            )
            unique_admissions = (
                self.data["hadm_id"].nunique() if "hadm_id" in self.data.columns else 0
            )
            unique_itemids = (
                self.data["itemid"].nunique() if "itemid" in self.data.columns else 0
            )
            logger.info(
                f"Final dataset: {unique_patients} patients, {unique_admissions} admissions, {unique_itemids} unique itemids"
            )

            # Check label coverage
            if "label" in self.data.columns:
                labeled_records = self.data["label"].notna().sum()
                label_coverage = labeled_records / len(self.data) * 100
                logger.info(
                    f"Label coverage: {labeled_records}/{len(self.data)} records ({label_coverage:.1f}%)"
                )
        else:
            logger.warning("❌ No lab data loaded - dataset is empty")

        # Convert time columns to datetime (both charttime and storetime)
        logger.info("Step 4: Converting time columns to datetime format...")
        time_columns = ["charttime", "storetime"]

        for time_col in time_columns:
            if time_col in self.data.columns:
                logger.info(f"Converting {time_col} to datetime...")
                try:
                    self.data[time_col] = pd.to_datetime(self.data[time_col])
                    logger.info(f"✅ {time_col} conversion successful")

                    # Log time range for this column
                    if not self.data.empty:
                        min_time = self.data[time_col].min()
                        max_time = self.data[time_col].max()
                        logger.info(f"{time_col} range: {min_time} to {max_time}")
                except Exception as e:
                    logger.error(f"❌ Error converting {time_col}: {e}")
            else:
                logger.warning(f"⚠️ {time_col} column not found in data")

        # Log information about units column
        if "valueuom" in self.data.columns:
            unique_units = self.data["valueuom"].nunique()
            logger.info(
                f"✅ Units column (valueuom) found: {unique_units} unique units"
            )

            # Log most common units for debugging
            if not self.data.empty:
                top_units = self.data["valueuom"].value_counts().head(10)
                logger.info(f"Top 10 most common units: {dict(top_units)}")
        else:
            logger.warning("⚠️ Units column (valueuom) not found in data")

        # Log critical columns for time series analysis
        critical_columns = ["charttime", "storetime", "valuenum", "valueuom"]
        available_columns = [
            col for col in critical_columns if col in self.data.columns
        ]
        missing_columns = [
            col for col in critical_columns if col not in self.data.columns
        ]

        logger.info(f"✅ Critical columns available: {available_columns}")
        if missing_columns:
            logger.warning(f"⚠️ Critical columns missing: {missing_columns}")

        # Explain the difference between charttime and storetime
        if "charttime" in self.data.columns and "storetime" in self.data.columns:
            logger.info("📋 Time column usage:")
            logger.info("  - charttime: When the measurement was taken (clinical time)")
            logger.info("  - storetime: When the result was stored in the system")
            logger.info(
                "  - For time series analysis, charttime should be used as primary temporal reference"
            )

        logger.info("✅ Lab data loading process completed successfully.")

        # Initialize simple itemid-to-label mapping
        self._initialize_itemid_mapping()

    def get_patient_data(
        self, patient_id: str, hadm_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get lab data for a specific patient and admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The hospital admission ID to filter by

        Returns:
            Dictionary containing the patient's lab data
        """
        if self.data is None:
            self.load_data()

        patient_data = self.data[self.data["subject_id"] == patient_id].copy()

        if hadm_id:
            patient_data = patient_data[patient_data["hadm_id"] == hadm_id]

        # Sort by charttime
        if "charttime" in patient_data.columns:
            patient_data = patient_data.sort_values("charttime")

        return {"lab_events": patient_data.to_dict("records")}

    def _initialize_itemid_mapping(self) -> None:
        """Initialize simple itemid-to-label mapping from d_labitems.csv."""
        try:
            logger.info("Initializing itemid-to-label mapping...")

            if self.lab_items is None or self.lab_items.empty:
                logger.warning("No lab items data available for mapping")
                return

            # Create simple mapping: itemid -> {label, fluid, category}
            self._itemid_to_label_map = {}
            for _, item in self.lab_items.iterrows():
                itemid = int(item["itemid"])
                self._itemid_to_label_map[itemid] = {
                    "label": str(item["label"]),
                    "fluid": str(item.get("fluid", "")),
                    "category": str(item.get("category", "")),
                }

            logger.info(
                f"Itemid mapping initialized: {len(self._itemid_to_label_map)} items mapped"
            )

            # Log some example mappings for key tests
            troponin_itemids = [51003, 52642, 51002]  # Known troponin itemids
            for itemid in troponin_itemids:
                if itemid in self._itemid_to_label_map:
                    mapping = self._itemid_to_label_map[itemid]
                    logger.info(
                        f"Troponin mapping: {itemid} -> {mapping['label']} (fluid: {mapping['fluid']})"
                    )

        except Exception as e:
            logger.error(f"Failed to initialize itemid mapping: {e}")
            self._itemid_to_label_map = None

    def get_lab_tests_by_itemids(
        self, patient_id: str, hadm_id: str, itemids: List[int]
    ) -> pd.DataFrame:
        """Get specific lab test results using itemids directly.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            itemids: List of itemids to search for

        Returns:
            DataFrame containing the requested test results with mapped labels
        """
        if self.data is None:
            self.load_data()

        logger.info(f"[{hadm_id}] Searching for tests using itemids: {itemids}")

        # Get tests using itemid-based search
        tests = self.data[
            (self.data["subject_id"] == patient_id)
            & (self.data["hadm_id"] == hadm_id)
            & (self.data["itemid"].isin(itemids))
        ].copy()

        logger.info(
            f"[{hadm_id}] Found {len(tests)} test records for itemids: {itemids}"
        )

        if not tests.empty:
            # Add mapped label information if available
            if self._itemid_to_label_map:
                for idx, test in tests.iterrows():
                    itemid = test.get("itemid")
                    if itemid in self._itemid_to_label_map:
                        mapping = self._itemid_to_label_map[itemid]
                        # Ensure we have the mapped label (in case merge didn't work)
                        if pd.isna(test.get("label")) or test.get("label") == "":
                            tests.at[idx, "label"] = mapping["label"]
                        if pd.isna(test.get("fluid")) or test.get("fluid") == "":
                            tests.at[idx, "fluid"] = mapping["fluid"]
                        if pd.isna(test.get("category")) or test.get("category") == "":
                            tests.at[idx, "category"] = mapping["category"]

            # Log details about found tests with units and time information
            for _, test in tests.iterrows():
                itemid = test.get("itemid", "N/A")
                label = test.get("label", "N/A")
                fluid = test.get("fluid", "N/A")
                value = test.get("valuenum", "N/A")
                unit = test.get("valueuom", "N/A")
                charttime = test.get("charttime", "N/A")
                storetime = test.get("storetime", "N/A")
                logger.info(
                    f"[{hadm_id}] Test found: itemid={itemid}, "
                    f"label='{label}', fluid='{fluid}', value={value} {unit}, "
                    f"charttime={charttime}, storetime={storetime}"
                )

            values = tests["valuenum"].dropna()
            if not values.empty:
                logger.info(f"[{hadm_id}] Values found: {list(values)}")
                logger.info(
                    f"[{hadm_id}] Value range: min={values.min():.6f}, max={values.max():.6f}"
                )

        # Sort by charttime
        if not tests.empty and "charttime" in tests.columns:
            tests = tests.sort_values("charttime")

        return tests

    def get_lab_test_info(self, itemid: int) -> Optional[Dict[str, str]]:
        """Get label information for a specific itemid.

        Args:
            itemid: The itemid to look up

        Returns:
            Dictionary with label, fluid, and category info, or None if not found
        """
        if self._itemid_to_label_map and itemid in self._itemid_to_label_map:
            return self._itemid_to_label_map[itemid]
        return None

    def search_itemids_by_label(self, search_term: str) -> List[int]:
        """Search for itemids that match a label search term.

        Args:
            search_term: Term to search for in labels

        Returns:
            List of matching itemids
        """
        if not self._itemid_to_label_map:
            return []

        matching_itemids = []
        search_term_lower = search_term.lower()

        for itemid, mapping in self._itemid_to_label_map.items():
            if search_term_lower in mapping["label"].lower():
                matching_itemids.append(itemid)

        return matching_itemids

    def get_troponin_tests(self, patient_id: str, hadm_id: str) -> pd.DataFrame:
        """Get troponin test results using direct itemid mapping.

        This method uses the simplified approach: direct itemid lookup with
        label mapping from d_labitems.csv.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            DataFrame containing troponin test results with proper labels
        """
        logger.info(f"[{hadm_id}] Getting troponin tests using direct itemid mapping")

        # Known troponin itemids from d_labitems.csv
        troponin_itemids = [51003, 52642, 51002]  # Troponin T, Troponin I, Troponin I

        # Use the simplified itemid-based approach
        troponin_tests = self.get_lab_tests_by_itemids(
            patient_id, hadm_id, troponin_itemids
        )

        if not troponin_tests.empty:
            logger.info(
                f"[{hadm_id}] Found {len(troponin_tests)} troponin tests using direct itemid mapping"
            )

            # Log troponin-specific analysis
            values = troponin_tests["valuenum"].dropna()
            if not values.empty:
                logger.info(f"[{hadm_id}] Troponin diagnostic threshold: 0.014 ng/mL")
                above_threshold = values[values > 0.014]
                logger.info(
                    f"[{hadm_id}] Values above threshold: {len(above_threshold)} out of {len(values)}"
                )
        else:
            logger.warning(
                f"[{hadm_id}] No troponin tests found using direct itemid mapping"
            )

        return troponin_tests

    def _get_troponin_tests_fallback(
        self, patient_id: str, hadm_id: str
    ) -> pd.DataFrame:
        """Fallback troponin test retrieval using hard-coded itemids.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            DataFrame containing troponin test results
        """
        if self.data is None:
            self.load_data()

        logger.info(f"[{hadm_id}] === TROPONIN DATA SEARCH DEBUG ===")
        logger.info(
            f"[{hadm_id}] Searching for patient_id='{patient_id}', hadm_id='{hadm_id}'"
        )
        logger.info(f"[{hadm_id}] Total lab records in dataset: {len(self.data)}")

        # Define troponin itemids from d_labitems.csv mapping
        # 51003: Troponin T, 52642: Troponin I, 51002: Troponin I
        troponin_itemids = [51003, 52642, 51002]
        logger.info(f"[{hadm_id}] Searching for troponin itemids: {troponin_itemids}")

        # Check what data exists for this patient/admission combination
        patient_admission_data = self.data[
            (self.data["subject_id"] == patient_id) & (self.data["hadm_id"] == hadm_id)
        ]
        logger.info(
            f"[{hadm_id}] Lab records for this patient/admission: {len(patient_admission_data)}"
        )

        if not patient_admission_data.empty:
            unique_itemids = patient_admission_data["itemid"].unique()
            logger.info(
                f"[{hadm_id}] Available itemids for this admission: {list(unique_itemids)[:20]}..."
            )  # Show first 20

            # Check if any troponin itemids are present
            troponin_itemids_found = [
                itemid for itemid in troponin_itemids if itemid in unique_itemids
            ]
            logger.info(
                f"[{hadm_id}] Troponin itemids found in data: {troponin_itemids_found}"
            )

            # Also check for labels (in case merge worked)
            unique_labels = patient_admission_data["label"].dropna().unique()
            troponin_labels = [
                label for label in unique_labels if "troponin" in str(label).lower()
            ]
            logger.info(f"[{hadm_id}] Troponin-related labels found: {troponin_labels}")
        else:
            logger.warning(
                f"[{hadm_id}] No lab data found for patient_id='{patient_id}', hadm_id='{hadm_id}'"
            )
            logger.info(
                f"[{hadm_id}] Sample patient_ids in dataset: {list(self.data['subject_id'].unique())[:5]}"
            )
            logger.info(
                f"[{hadm_id}] Sample hadm_ids in dataset: {list(self.data['hadm_id'].unique())[:5]}"
            )

        # Get troponin tests using itemid-based search (primary method)
        troponin_tests = self.data[
            (self.data["subject_id"] == patient_id)
            & (self.data["hadm_id"] == hadm_id)
            & (self.data["itemid"].isin(troponin_itemids))
        ].copy()

        # If no results with itemid, try label-based search as fallback
        if troponin_tests.empty:
            logger.info(
                f"[{hadm_id}] No troponin tests found by itemid, trying label-based search..."
            )
            troponin_tests = self.data[
                (self.data["subject_id"] == patient_id)
                & (self.data["hadm_id"] == hadm_id)
                & (self.data["label"].str.contains("troponin", case=False, na=False))
            ].copy()
            if not troponin_tests.empty:
                logger.info(
                    f"[{hadm_id}] Found troponin tests using label-based search"
                )

        logger.info(
            f"[{hadm_id}] Final troponin test records found: {len(troponin_tests)}"
        )

        if not troponin_tests.empty:
            # Log details about found tests
            for _, test in troponin_tests.iterrows():
                logger.info(
                    f"[{hadm_id}] Troponin test found: itemid={test.get('itemid', 'N/A')}, "
                    f"label='{test.get('label', 'N/A')}', value={test.get('valuenum', 'N/A')}"
                )

            values = troponin_tests["valuenum"].dropna()
            if not values.empty:
                logger.info(f"[{hadm_id}] Troponin values found: {list(values)}")
                logger.info(
                    f"[{hadm_id}] Troponin value range: min={values.min():.6f}, max={values.max():.6f}"
                )
                logger.info(f"[{hadm_id}] Diagnostic threshold: 0.014 ng/mL")
                above_threshold = values[values > 0.014]
                logger.info(
                    f"[{hadm_id}] Values above threshold: {len(above_threshold)} out of {len(values)}"
                )
            else:
                logger.warning(
                    f"[{hadm_id}] Troponin tests found but no numeric values available"
                )

        # Sort by charttime
        if not troponin_tests.empty and "charttime" in troponin_tests.columns:
            troponin_tests = troponin_tests.sort_values("charttime")

        return troponin_tests

    # Legacy method for backward compatibility
    def get_lab_tests_by_type(
        self, patient_id: str, hadm_id: str, test_type: str
    ) -> pd.DataFrame:
        """Legacy method - now uses direct itemid search with label mapping.

        Attempts to find itemids for the requested test type and uses direct lookup.
        """
        logger.warning(
            f"get_lab_tests_by_type is deprecated. Use get_lab_tests_by_itemids or search_itemids_by_label instead."
        )

        # Search for itemids that match the test type
        matching_itemids = self.search_itemids_by_label(test_type)
        if matching_itemids:
            logger.info(f"Found matching itemids for '{test_type}': {matching_itemids}")
            return self.get_lab_tests_by_itemids(patient_id, hadm_id, matching_itemids)

        # Fallback to label-based search
        logger.info(f"No itemids found for '{test_type}', using label search")
        return self.get_lab_tests_by_name(patient_id, hadm_id, test_type)

    def get_intelligent_categorization_summary(self) -> Dict[str, Any]:
        """Get a summary of the intelligent lab test categorization.

        Returns:
            Dictionary containing categorization statistics and details
        """
        if self._categorizer is None:
            return {"status": "not_initialized", "categories": 0, "total_tests": 0}

        return self._categorizer.get_categorization_summary()

    def search_lab_tests_by_criteria(
        self,
        patient_id: str,
        hadm_id: str,
        test_name: Optional[str] = None,
        fluid_type: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Advanced search for lab tests with multiple criteria.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            test_name: Partial test name to search for
            fluid_type: Specific fluid type (serum, plasma, urine, etc.)
            category_filter: Category filter from intelligent categorization

        Returns:
            DataFrame containing matching test results
        """
        if self.data is None:
            self.load_data()

        logger.info(
            f"[{hadm_id}] Advanced lab test search: name='{test_name}', fluid='{fluid_type}', category='{category_filter}'"
        )

        # Start with patient/admission filter
        tests = self.data[
            (self.data["subject_id"] == patient_id) & (self.data["hadm_id"] == hadm_id)
        ].copy()

        # Apply test name filter
        if test_name:
            tests = tests[tests["label"].str.contains(test_name, case=False, na=False)]

        # Apply fluid type filter
        if fluid_type:
            tests = tests[tests["fluid"].str.contains(fluid_type, case=False, na=False)]

        # Apply category filter using intelligent categorization
        if category_filter and self._categorizer:
            category_itemids = self._categorizer.get_itemids_for_category(
                category_filter
            )
            if category_itemids:
                tests = tests[tests["itemid"].isin(category_itemids)]

        logger.info(f"[{hadm_id}] Advanced search found {len(tests)} matching tests")

        # Sort by charttime
        if not tests.empty and "charttime" in tests.columns:
            tests = tests.sort_values("charttime")

        return tests

    def get_lab_tests_by_name(
        self, patient_id: str, hadm_id: str, test_name: str
    ) -> pd.DataFrame:
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
            (self.data["subject_id"] == patient_id)
            & (self.data["hadm_id"] == hadm_id)
            & (self.data["label"].str.contains(test_name, case=False, na=False))
        ].copy()

        if not tests.empty and "charttime" in tests.columns:
            tests = tests.sort_values("charttime")

        return tests

    def get_multiple_lab_tests(
        self, patient_id: str, hadm_id: str, test_types: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Get multiple lab test types for an admission using itemid mapping.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            test_types: List of test types from LAB_TEST_ITEMIDS keys

        Returns:
            Dictionary mapping test type to DataFrame of results
        """
        results = {}
        for test_type in test_types:
            results[test_type] = self.get_lab_tests_by_type(
                patient_id, hadm_id, test_type
            )
        return results

    def get_available_lab_test_types(self) -> List[str]:
        """Get all available lab test types that can be retrieved using itemid mapping.

        Returns:
            List of available test type keys
        """
        return list(self.LAB_TEST_ITEMIDS.keys())

    def get_comprehensive_lab_summary(
        self, patient_id: str, hadm_id: str
    ) -> Dict[str, Any]:
        """Get a comprehensive summary of all available lab tests for an admission.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing summary of all lab test categories
        """
        logger.info(
            f"[{hadm_id}] Generating comprehensive lab summary for patient {patient_id}"
        )

        summary = {
            "patient_id": patient_id,
            "hadm_id": hadm_id,
            "cardiac_markers": {},
            "metabolic_panel": {},
            "blood_count": {},
            "liver_function": {},
            "lipid_panel": {},
            "other_tests": {},
            "test_counts": {},
        }

        # Cardiac markers
        cardiac_tests = ["troponin", "ck_mb", "myoglobin"]
        for test in cardiac_tests:
            results = self.get_lab_tests_by_type(patient_id, hadm_id, test)
            summary["cardiac_markers"][test] = len(results)
            summary["test_counts"][test] = len(results)

        # Basic metabolic panel
        metabolic_tests = [
            "glucose",
            "sodium",
            "potassium",
            "chloride",
            "creatinine",
            "bun",
        ]
        for test in metabolic_tests:
            results = self.get_lab_tests_by_type(patient_id, hadm_id, test)
            summary["metabolic_panel"][test] = len(results)
            summary["test_counts"][test] = len(results)

        # Complete blood count
        cbc_tests = ["hemoglobin", "hematocrit", "white_blood_cells", "platelet_count"]
        for test in cbc_tests:
            results = self.get_lab_tests_by_type(patient_id, hadm_id, test)
            summary["blood_count"][test] = len(results)
            summary["test_counts"][test] = len(results)

        # Liver function
        liver_tests = [
            "alt",
            "ast",
            "alkaline_phosphatase",
            "bilirubin_total",
            "bilirubin_direct",
        ]
        for test in liver_tests:
            results = self.get_lab_tests_by_type(patient_id, hadm_id, test)
            summary["liver_function"][test] = len(results)
            summary["test_counts"][test] = len(results)

        # Lipid panel
        lipid_tests = [
            "cholesterol_total",
            "cholesterol_hdl",
            "cholesterol_ldl",
            "triglycerides",
        ]
        for test in lipid_tests:
            results = self.get_lab_tests_by_type(patient_id, hadm_id, test)
            summary["lipid_panel"][test] = len(results)
            summary["test_counts"][test] = len(results)

        # Other important tests
        other_tests = [
            "calcium",
            "magnesium",
            "albumin",
            "protein_total",
            "lactate",
            "hemoglobin_a1c",
        ]
        for test in other_tests:
            results = self.get_lab_tests_by_type(patient_id, hadm_id, test)
            summary["other_tests"][test] = len(results)
            summary["test_counts"][test] = len(results)

        total_tests = sum(summary["test_counts"].values())
        logger.info(
            f"[{hadm_id}] Lab summary complete: {total_tests} total lab tests found across all categories"
        )

        return summary

    def get_earliest_timestamp(
        self, patient_id: str, hadm_id: str
    ) -> Optional[pd.Timestamp]:
        """Get the earliest charttime for a given admission."""
        if self.data is None:
            self.load_data()

        admission_data = self.data[
            (self.data["subject_id"] == patient_id) & (self.data["hadm_id"] == hadm_id)
        ]

        if not admission_data.empty and "charttime" in admission_data.columns:
            return admission_data["charttime"].min()

        return None

    def get_all_admissions(self) -> List[Tuple[str, str]]:
        """Get all unique (patient_id, hadm_id) tuples from the lab data.

        Returns:
            A list of (patient_id, hadm_id) tuples.
        """
        if self.data is None:
            self.load_data()

        if "subject_id" not in self.data.columns or "hadm_id" not in self.data.columns:
            return []

        # Drop rows where subject_id or hadm_id is missing
        admissions_df = self.data[["subject_id", "hadm_id"]].dropna().drop_duplicates()
        # Convert to list of tuples
        admissions = [tuple(row) for row in admissions_df.itertuples(index=False)]
        return admissions
