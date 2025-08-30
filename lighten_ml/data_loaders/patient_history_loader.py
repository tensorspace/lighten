"""Patient history data loader for managing multi-visit analysis."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class PatientHistoryLoader(BaseDataLoader):
    """Load and process patient visit history from discharge notes."""

    def __init__(self, discharge_notes_path: str):
        super().__init__(discharge_notes_path)
        self.required_columns = ["subject_id", "hadm_id", "chartdate", "text"]
        self.patient_history = {}  # Cache for patient visit data

    def load_data(self) -> None:
        """Load discharge notes data with proper column handling."""
        logger.info(f"[DATA_LOADING] Loading patient history from {self.file_path}")
        logger.debug(f"[DEBUG] Required columns: {self.required_columns}")

        try:
            # Check file existence and size
            import os

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(
                    f"Patient history file not found: {self.file_path}"
                )

            file_size = os.path.getsize(self.file_path)
            logger.debug(f"[DEBUG] File size: {file_size:,} bytes")

            # Read the CSV file
            logger.debug(f"[DEBUG] Reading CSV file with patient history data")
            self.data = pd.read_csv(
                self.file_path,
                dtype={"subject_id": str, "hadm_id": str},
                parse_dates=(
                    ["chartdate"]
                    if "chartdate" in pd.read_csv(self.file_path, nrows=1).columns
                    else None
                ),
            )

            logger.debug(f"[DEBUG] Initial data shape: {self.data.shape}")
            logger.debug(f"[DEBUG] Columns found: {list(self.data.columns)}")

            # Validate required columns
            missing_cols = [
                col for col in self.required_columns if col not in self.data.columns
            ]
            if missing_cols:
                logger.error(f"[ERROR] Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            logger.debug(f"[DEBUG] All required columns present")

            # Check for null values in critical columns
            null_counts = {}
            for col in self.required_columns:
                null_count = self.data[col].isnull().sum()
                null_counts[col] = null_count
                if null_count > 0:
                    logger.warning(
                        f"[WARNING] Column '{col}' has {null_count} null values"
                    )

            # Sort by patient and date for chronological analysis
            logger.debug(f"[DEBUG] Sorting data by subject_id and chartdate")
            self.data = self.data.sort_values(["subject_id", "chartdate"])

            # Generate comprehensive statistics
            unique_patients = self.data["subject_id"].nunique()
            unique_admissions = self.data["hadm_id"].nunique()
            date_range_start = self.data["chartdate"].min()
            date_range_end = self.data["chartdate"].max()
            avg_visits_per_patient = (
                len(self.data) / unique_patients if unique_patients > 0 else 0
            )

            logger.info(
                f"[DATA_LOADED] Successfully loaded {len(self.data)} discharge note records"
            )
            logger.info(f"[DATA_STATS] Unique patients: {unique_patients}")
            logger.info(f"[DATA_STATS] Unique admissions: {unique_admissions}")
            logger.info(
                f"[DATA_STATS] Average visits per patient: {avg_visits_per_patient:.1f}"
            )
            logger.info(
                f"[DATA_STATS] Date range: {date_range_start} to {date_range_end}"
            )

            if date_range_start and date_range_end:
                total_days = (date_range_end - date_range_start).days
                logger.info(f"[DATA_STATS] Total span: {total_days} days")

            logger.debug(f"[DEBUG] Data loading completed successfully")

        except Exception as e:
            logger.error(f"[ERROR] Failed to load patient history data: {e}")
            logger.debug(f"[DEBUG] Exception details", exc_info=True)
            raise

    def get_patient_visit_history(self, patient_id: str) -> List[Dict]:
        """Get chronologically ordered visit history for a patient.

        Args:
            patient_id: Patient subject_id

        Returns:
            List of visit records ordered by date
        """
        logger.debug(f"[DEBUG] Getting visit history for patient {patient_id}")

        if self.data is None:
            logger.debug(
                f"[DEBUG] Data not loaded, loading now for patient {patient_id}"
            )
            self.load_data()

        # Check cache first
        if patient_id in self.patient_history:
            cached_visits = self.patient_history[patient_id]
            logger.debug(
                f"[DEBUG] Found cached visit history for patient {patient_id}: {len(cached_visits)} visits"
            )
            return cached_visits

        logger.debug(
            f"[DEBUG] No cached data found, querying database for patient {patient_id}"
        )

        # Filter patient data
        patient_data = self.data[self.data["subject_id"] == patient_id].copy()

        if patient_data.empty:
            logger.warning(f"[WARNING] No visit history found for patient {patient_id}")
            logger.debug(
                f"[DEBUG] Available patient IDs sample: {self.data['subject_id'].unique()[:5].tolist()}"
            )
            return []

        logger.debug(
            f"[DEBUG] Found {len(patient_data)} raw records for patient {patient_id}"
        )

        # Convert to list of dictionaries for easier processing
        visits = []
        for idx, row in patient_data.iterrows():
            visit = {
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "chartdate": row["chartdate"],
                "text": row["text"],
                "visit_order": len(visits) + 1,  # Sequential visit number
                "text_length": len(row["text"]) if pd.notna(row["text"]) else 0,
            }
            visits.append(visit)
            logger.debug(
                f"[DEBUG] Visit {len(visits)}: {row['hadm_id']} on {row['chartdate']} ({visit['text_length']} chars)"
            )

        # Cache the result
        self.patient_history[patient_id] = visits
        logger.debug(f"[DEBUG] Cached visit history for patient {patient_id}")

        # Log visit summary
        if visits:
            date_range = f"{visits[0]['chartdate']} to {visits[-1]['chartdate']}"
            total_text_length = sum(visit["text_length"] for visit in visits)
            logger.info(
                f"[VISIT_HISTORY] Patient {patient_id}: {len(visits)} visits from {date_range}"
            )
            logger.debug(
                f"[DEBUG] Total clinical text: {total_text_length:,} characters"
            )

        return visits

    def get_all_patient_ids(self) -> List[str]:
        """Get list of all unique patient IDs."""
        if self.data is None:
            self.load_data()

        patient_ids = self.data["subject_id"].unique().tolist()
        logger.info(f"Found {len(patient_ids)} unique patients")
        return patient_ids

    def get_patient_visit_summary(self, patient_id: str) -> Dict:
        """Get summary statistics for a patient's visit history.

        Args:
            patient_id: Patient subject_id

        Returns:
            Dictionary with visit summary statistics
        """
        visits = self.get_patient_visit_history(patient_id)

        if not visits:
            return {"total_visits": 0, "date_range": None, "admission_ids": []}

        dates = [visit["chartdate"] for visit in visits]
        admission_ids = [visit["hadm_id"] for visit in visits]

        return {
            "total_visits": len(visits),
            "date_range": {
                "first_visit": min(dates),
                "last_visit": max(dates),
                "span_days": (max(dates) - min(dates)).days if len(dates) > 1 else 0,
            },
            "admission_ids": admission_ids,
            "chronological_visits": [(v["hadm_id"], v["chartdate"]) for v in visits],
        }

    def get_visit_by_admission_id(
        self, patient_id: str, hadm_id: str
    ) -> Optional[Dict]:
        """Get specific visit record by admission ID.

        Args:
            patient_id: Patient subject_id
            hadm_id: Hospital admission ID

        Returns:
            Visit record or None if not found
        """
        visits = self.get_patient_visit_history(patient_id)

        for visit in visits:
            if visit["hadm_id"] == hadm_id:
                return visit

        logger.warning(f"Visit {hadm_id} not found for patient {patient_id}")
        return None

    def get_visits_before_date(
        self, patient_id: str, cutoff_date: datetime
    ) -> List[Dict]:
        """Get all visits for a patient before a specific date.

        Args:
            patient_id: Patient subject_id
            cutoff_date: Date cutoff for historical analysis

        Returns:
            List of visits before the cutoff date
        """
        visits = self.get_patient_visit_history(patient_id)

        historical_visits = [
            visit for visit in visits if visit["chartdate"] < cutoff_date
        ]

        logger.info(
            f"Found {len(historical_visits)} historical visits for patient {patient_id} before {cutoff_date}"
        )
        return historical_visits

    def get_visits_in_date_range(
        self, patient_id: str, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        """Get visits within a specific date range.

        Args:
            patient_id: Patient subject_id
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of visits within the date range
        """
        visits = self.get_patient_visit_history(patient_id)

        range_visits = [
            visit for visit in visits if start_date <= visit["chartdate"] <= end_date
        ]

        logger.info(
            f"Found {len(range_visits)} visits for patient {patient_id} between {start_date} and {end_date}"
        )
        return range_visits
