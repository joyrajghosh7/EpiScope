
import json
from pathlib import Path
from shapely.geometry import shape, mapping

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data" / "reference" / "bangladesh_districts.geojson"
OUTPUT = ROOT / "dashboard" / "static" / "data" / "bangladesh_districts.geojson"

print("Reading original district GeoJSON...")

with open(SOURCE, "r", encoding="utf-8") as file:
    geojson = json.load(file)

features = geojson["features"]
simplified_features = []

print("Original district count:", len(features))

for feature in features:
    geometry = shape(feature["geometry"])

    simplified = geometry.simplify(
        tolerance=0.003,
        preserve_topology=True
    )

    simplified_features.append({
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": mapping(simplified)
    })

result = {
    "type": "FeatureCollection",
    "features": simplified_features
}

if "crs" in geojson:
    result["crs"] = geojson["crs"]

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, separators=(",", ":"))

# Validate output
with open(OUTPUT, "r", encoding="utf-8") as file:
    check = json.load(file)

assert len(check["features"]) == 64, "District count mismatch!"

for feature in check["features"]:
    geometry = shape(feature["geometry"])
    assert not geometry.is_empty, "Empty district geometry!"
    assert geometry.is_valid, "Invalid district geometry!"

original_size = SOURCE.stat().st_size / (1024 * 1024)
new_size = OUTPUT.stat().st_size / (1024 * 1024)

print("\nMAP SIMPLIFICATION COMPLETED")
print("Districts:", len(check["features"]))
print(f"Original size: {original_size:.2f} MB")
print(f"Simplified size: {new_size:.2f} MB")
print(f"Size reduction: {(1 - new_size / original_size) * 100:.2f}%")
print("All 64 district geometries are valid.")
