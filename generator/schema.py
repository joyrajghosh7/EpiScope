
"""
EpiScope - Synthetic Data Generator
Data schema definitions for S4, S5 and S6.
"""

# Project configuration
START_YEAR = 2021
END_YEAR = 2025

DEFAULT_SEED = 42
DEFAULT_SCALE_FACTOR = 0.01

TOTAL_DISTRICTS = 64
MIN_FACILITIES = 600
MIN_OUTBREAKS = 30

# S4: Health-facility encounter records
ENCOUNTER_COLUMNS = [
    "encounter_id",
    "patient_id",
    "facility_id",
    "district",
    "onset_date",
    "report_date",
    "age_years",
    "sex",
    "icd10_code",
    "lab_confirmed",
    "admitted",
    "outcome"
]

# ICD-10 codes and associated diseases
DISEASE_CODES = {
    "Dengue": ["A90", "A91"],
    "Diarrhoea": ["A00", "A09"],
    "Influenza": ["J10", "J11"],
    "Malaria": ["B50", "B51", "B52", "B53", "B54"],
    "Other": ["Z00", "R50", "B99"]
}

# S5: Facility registry
FACILITY_COLUMNS = [
    "facility_id",
    "facility_name",
    "district",
    "facility_type",
    "latitude",
    "longitude",
    "beds",
    "icu_beds"
]

FACILITY_TYPES = [
    "Hospital",
    "Upazila Health Complex",
    "Clinic"
]

# S6: Daily stock and bed occupancy
STOCK_COLUMNS = [
    "snapshot_date",
    "facility_id",
    "iv_saline_stock",
    "ns1_kits_stock",
    "ors_stock",
    "antimalarial_stock",
    "occupied_beds"
]

# Separate outbreak ground truth.
# This file must NEVER be used as input to the detection algorithm.
GROUND_TRUTH_COLUMNS = [
    "outbreak_id",
    "disease",
    "district",
    "start_date",
    "end_date",
    "severity",
    "outbreak_type"
]

if __name__ == "__main__":
    print("EpiScope Generator Schema")
    print("Years:", START_YEAR, "-", END_YEAR)
    print("Districts:", TOTAL_DISTRICTS)
    print("Minimum facilities:", MIN_FACILITIES)
    print("Minimum outbreaks:", MIN_OUTBREAKS)
    print("Encounter fields:", len(ENCOUNTER_COLUMNS))
    print("Schema loaded successfully!")
