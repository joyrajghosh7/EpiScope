
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "reference"

POP_FILE = REF / "district_population.csv"
TOTAL_FILE = REF / "district_population_totals.csv"

print("=" * 60)
print("EPISCOPE - WORLDPOP POPULATION VALIDATION")
print("=" * 60)

population = pd.read_csv(POP_FILE)
totals = pd.read_csv(TOTAL_FILE)

errors = []

# 1. Record counts
print("\n1. RECORD COUNTS")
print("Age-sex records:", len(population))
print("District total records:", len(totals))

if len(population) != 1280:
    errors.append("Expected 1280 age-sex records.")

if len(totals) != 64:
    errors.append("Expected 64 district total records.")

# 2. District coverage
print("\n2. DISTRICT COVERAGE")

print("Age-sex districts:", population["district_code"].nunique())
print("Total districts:", totals["district_code"].nunique())

if population["district_code"].nunique() != 64:
    errors.append("Age-sex data does not contain 64 districts.")

if totals["district_code"].nunique() != 64:
    errors.append("District totals do not contain 64 districts.")

# 3. Age groups per district
print("\n3. AGE GROUP COUNTS")

age_counts = population.groupby("district_code").size()

print("Minimum age groups:", age_counts.min())
print("Maximum age groups:", age_counts.max())

if not (age_counts == 20).all():
    errors.append("Some districts do not have exactly 20 age groups.")

# 4. Duplicate records
print("\n4. DUPLICATE CHECKS")

age_duplicates = population.duplicated(
    subset=["district_code", "age_class"]
).sum()

total_duplicates = totals.duplicated(
    subset=["district_code"]
).sum()

print("Duplicate district-age records:", age_duplicates)
print("Duplicate district totals:", total_duplicates)

if age_duplicates > 0:
    errors.append("Duplicate district-age records found.")

if total_duplicates > 0:
    errors.append("Duplicate district totals found.")

# 5. Missing values
print("\n5. MISSING VALUE CHECK")

print("Missing age-sex values:", population.isnull().sum().sum())
print("Missing total values:", totals.isnull().sum().sum())

if population.isnull().any().any():
    errors.append("Missing values found in age-sex dataset.")

if totals.isnull().any().any():
    errors.append("Missing values found in totals dataset.")

# 6. Numeric validation
print("\n6. NUMERIC VALIDATION")

numeric_columns = [
    "male_population",
    "female_population",
    "total_population",
    "area_km2"
]

for column in numeric_columns:
    population[column] = pd.to_numeric(
        population[column], errors="coerce"
    )

total_numeric_columns = [
    "total_population",
    "area_km2"
]

for column in total_numeric_columns:
    totals[column] = pd.to_numeric(
        totals[column], errors="coerce"
    )

if population[numeric_columns].isnull().any().any():
    errors.append("Invalid numeric values in age-sex dataset.")

if totals[total_numeric_columns].isnull().any().any():
    errors.append("Invalid numeric values in totals dataset.")

if (population[numeric_columns] < 0).any().any():
    errors.append("Negative values in age-sex dataset.")

if (totals[total_numeric_columns] < 0).any().any():
    errors.append("Negative values in totals dataset.")

# 7. Check age-sex totals within each row
print("\n7. AGE-SEX ROW CONSISTENCY")

difference = (
    population["male_population"]
    + population["female_population"]
    - population["total_population"]
).abs()

print("Maximum row difference:", difference.max())

if (difference > 0.02).any():
    errors.append("Male + female does not match row total.")

# 8. Year validation
print("\n8. YEAR VALIDATION")

print("Age-sex years:", population["year"].unique())
print("Total years:", totals["year"].unique())

if not (population["year"] == 2022).all():
    errors.append("Unexpected year in age-sex dataset.")

if not (totals["year"] == 2022).all():
    errors.append("Unexpected year in district totals.")

# 9. Dataset source
print("\n9. DATA SOURCE")

print("Age-sex sources:", population["data_source"].unique())
print("Total sources:", totals["data_source"].unique())

# 10. District consistency
print("\n10. DISTRICT CONSISTENCY")

population_codes = set(population["district_code"])
total_codes = set(totals["district_code"])

if population_codes != total_codes:
    errors.append("District codes differ between the two datasets.")

# 11. District population sample
print("\n11. SAMPLE DISTRICT POPULATIONS")

print(
    totals[
        ["district_code", "district", "total_population"]
    ].head(10).to_string(index=False)
)

# IMPORTANT:
# We deliberately do NOT compare the sum of all 20 age bands
# against total_population because the returned upper age
# categories may overlap.

print("\n" + "=" * 60)

if errors:
    print("VALIDATION FAILED")

    for error in errors:
        print("ERROR:", error)

else:
    print("POPULATION STRUCTURAL VALIDATION PASSED!")

print("=" * 60)
