import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE = PROJECT_ROOT / "data" / "reference" / "bangladesh_districts.geojson"
OUTPUT = PROJECT_ROOT / "data" / "reference" / "districts.csv"

with open(SOURCE, "r", encoding="utf-8") as file:
    geojson = json.load(file)

districts = []

for feature in geojson["features"]:
    properties = feature["properties"]

    districts.append({
        "district_code": properties["adm2_pcode"],
        "district": properties["adm2_en"],
        "division_code": properties["adm1_pcode"],
        "division": properties["adm1_en"]
    })

districts.sort(key=lambda x: x["district"])

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "district_code",
            "district",
            "division_code",
            "division"
        ]
    )

    writer.writeheader()
    writer.writerows(districts)

print("District reference CSV created!")
print("Total districts:", len(districts))
print("Output:", OUTPUT)

print("\nDistricts by division:")

division_counts = {}

for district in districts:
    division = district["division"]
    division_counts[division] = division_counts.get(division, 0) + 1

for division, count in sorted(division_counts.items()):
    print(f"{division}: {count}")