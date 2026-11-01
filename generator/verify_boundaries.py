import json
from pathlib import Path

path = Path("data/reference/bangladesh_districts.geojson")

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

features = data["features"]

print("Total districts:", len(features))
print("\nAvailable property fields:")

properties = features[0]["properties"]

for key in properties:
    print("-", key)

print("\nFirst 10 district records:")

for feature in features[:10]:
    props = feature["properties"]
    print(props)

print("\nGeometry type:", features[0]["geometry"]["type"])
print("\nBoundary verification completed!")