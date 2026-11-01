import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://gis.dghs.gov.bd/server/rest/services/Hosted/"
    "bgd_admbnda_adm2_bbs_20201113/FeatureServer/0/query"
)

params = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson"
}

print("Downloading Bangladesh district boundaries from DGHS...")

response = requests.get(URL, params=params, timeout=90)
response.raise_for_status()

geojson = response.json()

if "features" not in geojson:
    raise ValueError(f"Unexpected response: {geojson}")

features = geojson["features"]

if len(features) != 64:
    raise ValueError(f"Expected 64 districts, received {len(features)}")

output = REFERENCE_DIR / "bangladesh_districts.geojson"

with open(output, "w", encoding="utf-8") as file:
    json.dump(geojson, file, ensure_ascii=False)

print("\nSUCCESS!")
print("District boundaries:", len(features))
print("Saved to:", output)
print("Coordinate system: EPSG:4326")
print("Source: Bangladesh DGHS GIS")