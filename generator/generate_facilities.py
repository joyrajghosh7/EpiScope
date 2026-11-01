
import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "bangladesh_districts.geojson"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "facilities"
    / "facilities.csv"
)

SEED = 42
FACILITIES_PER_DISTRICT = 10

rng = random.Random(SEED)


def point_in_ring(lon, lat, ring):
    """Check whether a point is inside a polygon ring."""

    inside = False

    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]

        if (y1 > lat) != (y2 > lat):
            intersection = (
                (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            )

            if lon < intersection:
                inside = not inside

    return inside


def point_in_polygon(lon, lat, polygon):
    """Check polygon exterior and exclude its holes."""

    exterior = polygon[0]

    if not point_in_ring(lon, lat, exterior):
        return False

    for hole in polygon[1:]:
        if point_in_ring(lon, lat, hole):
            return False

    return True


def random_point_in_geometry(geometry):
    """Generate a random point inside Polygon or MultiPolygon."""

    geometry_type = geometry["type"]

    if geometry_type == "Polygon":
        polygons = [geometry["coordinates"]]

    elif geometry_type == "MultiPolygon":
        polygons = geometry["coordinates"]

    else:
        raise ValueError(
            f"Unsupported geometry type: {geometry_type}"
        )

    for _ in range(10000):

        polygon = rng.choice(polygons)
        exterior = polygon[0]

        min_lon = min(point[0] for point in exterior)
        max_lon = max(point[0] for point in exterior)

        min_lat = min(point[1] for point in exterior)
        max_lat = max(point[1] for point in exterior)

        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)

        if point_in_polygon(lon, lat, polygon):
            return round(lat, 6), round(lon, 6)

    raise RuntimeError(
        "Could not find a point inside district geometry."
    )


def generate_facilities():

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_FILE}"
        )

    with open(BOUNDARY_FILE, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    facilities = []
    facility_number = 1

    for feature in geojson["features"]:

        props = feature["properties"]

        district = props["adm2_en"]
        district_code = props["adm2_pcode"]
        geometry = feature["geometry"]

        for local_number in range(
            1, FACILITIES_PER_DISTRICT + 1
        ):

            latitude, longitude = random_point_in_geometry(
                geometry
            )

            facility_type = rng.choices(
                [
                    "Hospital",
                    "Upazila Health Complex",
                    "Clinic"
                ],
                weights=[20, 35, 45],
                k=1
            )[0]

            if facility_type == "Hospital":
                beds = rng.randint(100, 500)

            elif facility_type == "Upazila Health Complex":
                beds = rng.randint(20, 100)

            else:
                beds = rng.randint(5, 40)

            icu_beds = rng.randint(
                0, max(1, beds // 10)
            )

            facilities.append({
                "facility_id": f"FAC{facility_number:05d}",
                "facility_name": (
                    f"{district} {facility_type} "
                    f"{local_number:02d}"
                ),
                "district": district,
                "district_code": district_code,
                "facility_type": facility_type,
                "latitude": latitude,
                "longitude": longitude,
                "beds": beds,
                "icu_beds": icu_beds
            })

            facility_number += 1

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=facilities[0].keys()
        )

        writer.writeheader()
        writer.writerows(facilities)

    print("\nFacility registry generated successfully!")
    print("Total facilities:", len(facilities))
    print("Districts covered:", len(geojson["features"]))
    print(
        "Facilities per district:",
        FACILITIES_PER_DISTRICT
    )
    print("Output:", OUTPUT_FILE)

    print("\nFirst 5 facilities:")

    for facility in facilities[:5]:
        print(facility)


if __name__ == "__main__":
    generate_facilities()
