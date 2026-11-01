
import csv
from pathlib import Path
from collections import Counter
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "weather"
    / "weather_daily.csv"
)

EXPECTED_DISTRICTS = 64
EXPECTED_DAYS = 1826
EXPECTED_RECORDS = EXPECTED_DISTRICTS * EXPECTED_DAYS

REQUIRED_COLUMNS = [
    "district_code",
    "district",
    "date",
    "latitude",
    "longitude",
    "precipitation_mm",
    "temperature_mean_c",
    "humidity_mean_percent",
    "weather_source"
]

print("=" * 55)
print("EPISCOPE - WEATHER DATA VALIDATION")
print("=" * 55)

with open(WEATHER_FILE, encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    records = list(reader)
    columns = reader.fieldnames

print("\n1. FILE AND COLUMN VALIDATION")

print("File exists:", WEATHER_FILE.exists())
print("Required columns present:",
      all(column in columns for column in REQUIRED_COLUMNS))

print("\n2. RECORD COUNT")

print("Total records:", len(records))
print("Expected records:", EXPECTED_RECORDS)
print("Record count correct:", len(records) == EXPECTED_RECORDS)

print("\n3. DISTRICT COVERAGE")

district_counts = Counter(
    row["district_code"] for row in records
)

print("Unique districts:", len(district_counts))

incorrect_districts = {
    code: count
    for code, count in district_counts.items()
    if count != EXPECTED_DAYS
}

print("Districts with incorrect daily counts:",
      len(incorrect_districts))

if incorrect_districts:
    for code, count in incorrect_districts.items():
        print(code, ":", count)

print("\n4. DATE VALIDATION")

invalid_dates = 0
date_values = set()

for row in records:
    try:
        date_values.add(date.fromisoformat(row["date"]))
    except ValueError:
        invalid_dates += 1

print("Unique dates:", len(date_values))
print("Invalid dates:", invalid_dates)

if date_values:
    print("First date:", min(date_values))
    print("Last date:", max(date_values))

print("\n5. DUPLICATE VALIDATION")

district_date_pairs = [
    (row["district_code"], row["date"])
    for row in records
]

duplicate_pairs = (
    len(district_date_pairs) - len(set(district_date_pairs))
)

print("Duplicate district-date records:", duplicate_pairs)

print("\n6. WEATHER VALUE VALIDATION")

missing_values = Counter()
invalid_values = Counter()

for row in records:

    fields = [
        "precipitation_mm",
        "temperature_mean_c",
        "humidity_mean_percent"
    ]

    values = {}

    for field in fields:
        value = row[field].strip()

        if value == "":
            missing_values[field] += 1
            continue

        try:
            values[field] = float(value)
        except ValueError:
            invalid_values[field] += 1

    if "precipitation_mm" in values:
        if values["precipitation_mm"] < 0:
            invalid_values["negative_precipitation"] += 1

    if "temperature_mean_c" in values:
        if not -10 <= values["temperature_mean_c"] <= 55:
            invalid_values["temperature_out_of_range"] += 1

    if "humidity_mean_percent" in values:
        if not 0 <= values["humidity_mean_percent"] <= 100:
            invalid_values["humidity_out_of_range"] += 1

print("Missing weather values:")

for field in fields:
    print(field, ":", missing_values[field])

print("Invalid weather values:")

if invalid_values:
    for field, count in invalid_values.items():
        print(field, ":", count)
else:
    print("None")

print("\n7. SOURCE VALIDATION")

sources = Counter(
    row["weather_source"] for row in records
)

for source, count in sources.items():
    print(source, ":", count)

print("\n8. FINAL RESULT")

checks = [
    len(records) == EXPECTED_RECORDS,
    len(district_counts) == EXPECTED_DISTRICTS,
    not incorrect_districts,
    invalid_dates == 0,
    len(date_values) == EXPECTED_DAYS,
    duplicate_pairs == 0,
    not invalid_values,
    all(column in columns for column in REQUIRED_COLUMNS)
]

if all(checks):
    print("STRUCTURAL VALIDATION PASSED!")
    if any(missing_values.values()):
        print("WARNING: Some weather measurements are missing.")
        print("We must handle missing values before analysis.")
    else:
        print("All weather measurements are present.")
else:
    print("VALIDATION FAILED - CHECK THE RESULTS ABOVE.")

print("=" * 55)
