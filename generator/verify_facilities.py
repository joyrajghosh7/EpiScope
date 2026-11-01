
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "facilities"
    / "facilities.csv"
)

with open(FILE, "r", encoding="utf-8-sig") as file:
    facilities = list(csv.DictReader(file))

print("===== FACILITY DATA VALIDATION =====")

# Total records
print("Total facilities:", len(facilities))

# Unique IDs
ids = [f["facility_id"] for f in facilities]
print("Unique facility IDs:", len(set(ids)))

# District coverage
districts = set(f["district"] for f in facilities)
print("Districts covered:", len(districts))

# Facilities per district
district_counts = Counter(
    f["district"] for f in facilities
)

incorrect = {
    district: count
    for district, count in district_counts.items()
    if count != 10
}

print("Districts with incorrect facility counts:", len(incorrect))

# Facility types
print("\nFacility types:")

type_counts = Counter(
    f["facility_type"] for f in facilities
)

for facility_type, count in type_counts.items():
    print(f"{facility_type}: {count}")

# Invalid values
invalid = []

for facility in facilities:
    if float(facility["latitude"]) < 20 or float(facility["latitude"]) > 27:
        invalid.append(facility["facility_id"])

    if float(facility["longitude"]) < 88 or float(facility["longitude"]) > 93:
        invalid.append(facility["facility_id"])

    if int(facility["icu_beds"]) > int(facility["beds"]):
        invalid.append(facility["facility_id"])

print("\nInvalid records:", len(set(invalid)))

if (
    len(facilities) == 640
    and len(set(ids)) == 640
    and len(districts) == 64
    and len(incorrect) == 0
    and len(invalid) == 0
):
    print("\nVALIDATION PASSED!")
else:
    print("\nVALIDATION FAILED!")
