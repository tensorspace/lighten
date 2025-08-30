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
            logger.info(f"[SUCCESS] Lab items loaded: {len(self.lab_items)} items")
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
            logger.error(f"[ERROR] Lab items file not found at {self.d_labitems_path}")
            self.lab_items = pd.DataFrame()
        except Exception as e:
            logger.error(f"[ERROR] Error loading lab items: {e}")
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
            logger.error(f"[ERROR] Lab events file not found: {self.lab_events_path}")
            self.data = pd.DataFrame()
            return
        except Exception as e:
            logger.error(f"[ERROR] Error loading lab events: {e}")
            self.data = pd.DataFrame()
            return

        # Concatenate all chunks
        logger.info(f"Step 3: Concatenating {len(chunks)} chunks...")
        self.data = pd.concat(chunks) if chunks else pd.DataFrame()

        if not self.data.empty:
            logger.info(
                f"[SUCCESS] Lab events concatenated: {len(self.data)} total records"
            )

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
            logger.warning("[WARNING] No lab data loaded - dataset is empty")

        # Convert time columns to datetime (both charttime and storetime)
        logger.info("Step 4: Converting time columns to datetime format...")
        time_columns = ["charttime", "storetime"]

        for time_col in time_columns:
            if time_col in self.data.columns:
                logger.info(f"Converting {time_col} to datetime...")
                try:
                    self.data[time_col] = pd.to_datetime(self.data[time_col])
                    logger.info(f"[SUCCESS] {time_col} conversion successful")

                    # Log time range for this column
                    if not self.data.empty:
                        min_time = self.data[time_col].min()
                        max_time = self.data[time_col].max()
                        logger.info(f"{time_col} range: {min_time} to {max_time}")
                except Exception as e:
                    logger.error(f"[ERROR] Error converting {time_col}: {e}")
            else:
                logger.warning(f"[WARNING] {time_col} column not found in data")

        # Log information about units column
        if "valueuom" in self.data.columns:
            unique_units = self.data["valueuom"].nunique()
            logger.info(
                f"[SUCCESS] Units column (valueuom) found: {unique_units} unique units"
            )

            # Log most common units for debugging
            if not self.data.empty:
                top_units = self.data["valueuom"].value_counts().head(10)
                logger.info(f"Top 10 most common units: {dict(top_units)}")
        else:
            logger.warning("[WARNING] Units column (valueuom) not found in data")

        # Log critical columns for time series analysis
        critical_columns = ["charttime", "storetime", "valuenum", "valueuom"]
        available_columns = [
            col for col in critical_columns if col in self.data.columns
        ]
        missing_columns = [
            col for col in critical_columns if col not in self.data.columns
        ]

        logger.info(f"[SUCCESS] Critical columns available: {available_columns}")
        if missing_columns:
            logger.warning(f"[WARNING] Critical columns missing: {missing_columns}")

        # Explain the difference between charttime and storetime
        if "charttime" in self.data.columns and "storetime" in self.data.columns:
            logger.info("[INFO] Time column usage:")
            logger.info("  - charttime: When the measurement was taken (clinical time)")
            logger.info("  - storetime: When the result was stored in the system")
            logger.info(
                "  - For time series analysis, charttime should be used as primary temporal reference"
            )

        logger.info("[SUCCESS] Lab data loading process completed successfully.")

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
            troponin_itemids = [
                51003
            ]  # Troponin T ONLY - clinical guideline compliance
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
        self, patient_id: str, itemids: List[int], hadm_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get specific lab test results for a patient using itemids.

        Args:
            patient_id: The ID of the patient.
            itemids: List of itemids to search for.
            hadm_id: Optional hospital admission ID to filter by.

        Returns:
            DataFrame containing the requested test results.
        """
        if self.data is None:
            self.load_data()

        log_prefix = f"[{patient_id}]"
        if hadm_id:
            log_prefix += f"[{hadm_id}]"
        log_prefix += " [LAB_LOADER]"

        logger.info(f"{log_prefix} Searching for tests with itemids: {itemids}")

        # Filter by patient and itemids first
        patient_data = self.data[
            (self.data["subject_id"] == patient_id)
            & (self.data["itemid"].isin(itemids))
        ].copy()

        # Optionally filter by admission
        if hadm_id:
            tests = patient_data[patient_data["hadm_id"] == hadm_id].copy()
        else:
            tests = patient_data

        logger.info(f"{log_prefix} Found {len(tests)} records for itemids: {itemids}")

        if not tests.empty:
            values = tests["valuenum"].dropna()
            if not values.empty:
                logger.debug(
                    f"{log_prefix} Value range: min={values.min():.4f}, max={values.max():.4f}"
                )

        # Sort by charttime for chronological order
        if not tests.empty and "charttime" in tests.columns:
            tests = tests.sort_values("charttime")

        return tests

    def get_troponin_tests(
        self, patient_id: str, hadm_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get all Troponin T tests for a patient.

        Args:
            patient_id: The ID of the patient.
            hadm_id: Optional hospital admission ID to filter by.

        Returns:
            DataFrame containing Troponin T test results.
        """
        # Per clinical guidelines, only Troponin T (itemid 51003) is used.
        troponin_itemids = [51003]
        return self.get_lab_tests_by_itemids(patient_id, troponin_itemids, hadm_id)

    def get_patient_troponin_history(self, patient_id: str) -> pd.DataFrame:
        """Get the complete Troponin T history for a patient across all admissions.

        Args:
            patient_id: The ID of the patient.

        Returns:
            DataFrame containing all Troponin T tests for the patient.
        """
        logger.info(
            f"[{patient_id}] [LAB_LOADER] Fetching complete troponin history..."
        )
        troponin_history = self.get_troponin_tests(patient_id, hadm_id=None)
        logger.info(
            f"[{patient_id}] [LAB_LOADER] Found {len(troponin_history)} total troponin records."
        )
        return troponin_history

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

    def get_patient_troponin_history(self, patient_id: str) -> pd.DataFrame:
        """Get complete troponin history for a patient across all admissions.

        Args:
            patient_id: Patient subject_id

        Returns:
            DataFrame with all troponin tests for the patient, sorted chronologically
        """
        logger.info(
            f"[{patient_id}] [LAB_LOADER] Fetching complete troponin history..."
        )
        troponin_history = self.get_troponin_tests(patient_id, hadm_id=None)
        logger.info(
            f"[{patient_id}] [LAB_LOADER] Found {len(troponin_history)} total troponin records."
        )
        return troponin_history

    def get_patient_lab_history(
        self, patient_id: str, itemids: List[int]
    ) -> pd.DataFrame:
        """Get complete lab test history for a patient across all admissions.

        Args:
            patient_id: Patient subject_id
            itemids: List of lab test itemids to retrieve

        Returns:
            DataFrame with all specified lab tests for the patient, sorted chronologically
        """
        logger.info(
            f"[{patient_id}] [LAB_LOADER] Getting lab history for itemids: {itemids}"
        )
        logger.debug(
            f"[{patient_id}] [DEBUG] Starting patient-level lab data collection for {len(itemids)} item types"
        )

        if self.data is None:
            logger.debug(f"[{patient_id}] [DEBUG] Lab data not loaded, loading now")
            self.load_data()

        # Get all lab tests for this patient
        logger.debug(f"[{patient_id}] [DEBUG] Filtering lab data for patient")
        patient_data = self.data[self.data["subject_id"] == patient_id]

        if patient_data.empty:
            logger.warning(f"[{patient_id}] [WARNING] No lab data found for patient")
            logger.debug(
                f"[{patient_id}] [DEBUG] Available patient IDs sample: {self.data['subject_id'].unique()[:5].tolist()}"
            )
            return pd.DataFrame()

        logger.debug(
            f"[{patient_id}] [DEBUG] Found {len(patient_data)} total lab records"
        )
        logger.debug(
            f"[{patient_id}] [DEBUG] Available itemids in patient data: {sorted(patient_data['itemid'].unique())[:20]}"
        )

        # Filter for specified itemids
        logger.debug(
            f"[{patient_id}] [DEBUG] Filtering for requested itemids: {itemids}"
        )
        lab_data = patient_data[patient_data["itemid"].isin(itemids)]

        if lab_data.empty:
            logger.info(
                f"[{patient_id}] [LAB_RESULT] No lab tests found for itemids {itemids}"
            )
            logger.debug(
                f"[{patient_id}] [DEBUG] Requested itemids not found in patient data"
            )
            return pd.DataFrame()

        # Sort chronologically by charttime
        logger.debug(f"[{patient_id}] [DEBUG] Sorting lab data chronologically")
        lab_data = lab_data.sort_values("charttime")

        # Generate statistics by itemid
        unique_admissions = lab_data["hadm_id"].nunique()
        date_range_start = lab_data["charttime"].min()
        date_range_end = lab_data["charttime"].max()

        logger.info(
            f"[{patient_id}] [LAB_RESULT] Found {len(lab_data)} lab tests across {unique_admissions} admissions"
        )
        logger.info(
            f"[{patient_id}] [LAB_RESULT] Date range: {date_range_start} to {date_range_end}"
        )

        # Log breakdown by itemid
        itemid_counts = lab_data["itemid"].value_counts()
        for itemid, count in itemid_counts.items():
            logger.debug(f"[{patient_id}] [DEBUG] ItemID {itemid}: {count} tests")

        logger.debug(f"[{patient_id}] [DEBUG] Lab history collection completed")
        return lab_data

    def get_all_patient_ids(self) -> List[str]:
        """Get list of all unique patient IDs in the lab data.

        Returns:
            List of unique patient subject_ids
        """
        logger.debug(f"[DEBUG] Getting all unique patient IDs from lab data")

        if self.data is None:
            logger.debug(f"[DEBUG] Lab data not loaded, loading now")
            self.load_data()

        if "subject_id" not in self.data.columns:
            logger.error(f"[ERROR] No subject_id column found in lab data")
            logger.debug(f"[DEBUG] Available columns: {list(self.data.columns)}")
            return []

        logger.debug(
            f"[DEBUG] Extracting unique patient IDs from {len(self.data)} lab records"
        )
        patient_ids = self.data["subject_id"].dropna().unique().tolist()

        logger.info(
            f"[PATIENT_IDS] Found {len(patient_ids)} unique patients in lab data"
        )
        logger.debug(f"[DEBUG] Sample patient IDs: {patient_ids[:5]}")

        return patient_ids

    def get_patient_admission_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get summary of all admissions for a patient with lab data.

        Args:
            patient_id: Patient subject_id

        Returns:
            Dictionary with admission summary statistics
        """
        if self.data is None:
            self.load_data()

        patient_data = self.data[self.data["subject_id"] == patient_id]

        if patient_data.empty:
            return {
                "patient_id": patient_id,
                "total_admissions": 0,
                "total_lab_tests": 0,
                "date_range": None,
                "admission_ids": [],
            }

        admission_ids = patient_data["hadm_id"].dropna().unique().tolist()
        total_tests = len(patient_data)

        # Get date range
        date_range = None
        if "charttime" in patient_data.columns:
            valid_dates = patient_data["charttime"].dropna()
            if not valid_dates.empty:
                date_range = {
                    "first_test": valid_dates.min(),
                    "last_test": valid_dates.max(),
                    "span_days": (valid_dates.max() - valid_dates.min()).days,
                }

        summary = {
            "patient_id": patient_id,
            "total_admissions": len(admission_ids),
            "total_lab_tests": total_tests,
            "date_range": date_range,
            "admission_ids": admission_ids,
        }

        logger.info(
            f"[{patient_id}] Patient summary: {summary['total_admissions']} admissions, {summary['total_lab_tests']} lab tests"
        )

        return summary
