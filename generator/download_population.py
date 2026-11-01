
import csv
import json
import time
import requests

from pathlib import Path
from shapely.geometry import shape, mapping
from shapely import make_valid


# ============================================================
# EPISCOPE - WORLDPOP POPULATION DOWNLOADER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    PROJECT_ROOT / "data" / "reference"
    / "bangladesh_districts.geojson"
)

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
RAW_DIR = REFERENCE_DIR / "worldpop_raw"

OUTPUT_FILE = REFERENCE_DIR / "district_population.csv"
TOTALS_FILE = REFERENCE_DIR / "district_population_totals.csv"

API_URL = "https://api.worldpop.org/v2"

YEAR = 2022
RESOLUTION = "1km"
SEX = "both"

POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 120
MAX_RETRIES = 5

# Keep the request comfortably below common server body limits.
MAX_GEOMETRY_BYTES = 700_000

# Simplification tolerances in degrees.
SIMPLIFICATION_TOLERANCES = [
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02
]


# ============================================================
# HTTP REQUEST WITH RETRIES
# ============================================================

def request_with_retry(method, url, **kwargs):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.request(
                method,
                url,
                timeout=180,
                **kwargs
            )

            # A 413 is not a temporary network error.
            if response.status_code == 413:
                raise RuntimeError(
                    "WorldPop rejected the request with HTTP 413. "
                    "The submitted geometry is still too large. "
                    "Check the geometry simplification."
                )

            if response.status_code == 429:

                wait_seconds = 60 * attempt

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:

            print(f"Request error: {error}")

            if attempt == MAX_RETRIES:
                raise

            wait_seconds = 15 * attempt

            print(
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Maximum API retries exceeded.")


# ============================================================
# PREPARE AND SIMPLIFY LARGE DISTRICT GEOMETRY
# ============================================================

def prepare_geometry(original_geometry, district_name):

    original_size = len(
        json.dumps(
            original_geometry,
            separators=(",", ":")
        ).encode("utf-8")
    )

    print(
        f"Original geometry size: "
        f"{original_size / 1024:.1f} KB"
    )

    # Keep small geometries unchanged.
    if original_size <= MAX_GEOMETRY_BYTES:

        print("Geometry size acceptable; no simplification needed.")

        return original_geometry

    print(
        "Large geometry detected. "
        "Simplifying district boundary..."
    )

    geom = shape(original_geometry)

    if not geom.is_valid:

        print("Repairing invalid geometry...")

        geom = make_valid(geom)

    if geom.geom_type not in ("Polygon", "MultiPolygon"):

        raise ValueError(
            f"Unsupported geometry type for {district_name}: "
            f"{geom.geom_type}"
        )

    for tolerance in SIMPLIFICATION_TOLERANCES:

        simplified = geom.simplify(
            tolerance,
            preserve_topology=True
        )

        if not simplified.is_valid:

            simplified = make_valid(simplified)

        if simplified.is_empty:

            continue

        if simplified.geom_type not in (
            "Polygon",
            "MultiPolygon"
        ):

            continue

        simplified_geojson = mapping(simplified)

        simplified_size = len(
            json.dumps(
                simplified_geojson,
                separators=(",", ":")
            ).encode("utf-8")
        )

        print(
            f"Tolerance {tolerance}: "
            f"{simplified_size / 1024:.1f} KB"
        )

        if simplified_size <= MAX_GEOMETRY_BYTES:

            print(
                f"Geometry simplified successfully for "
                f"{district_name}."
            )

            print(
                f"Size reduced from "
                f"{original_size / 1024:.1f} KB to "
                f"{simplified_size / 1024:.1f} KB"
            )

            return simplified_geojson

    raise ValueError(
        f"Could not reduce geometry for {district_name} "
        f"below {MAX_GEOMETRY_BYTES / 1024:.0f} KB. "
        "A different API submission strategy is required."
    )


# ============================================================
# LOAD DISTRICT BOUNDARIES
# ============================================================

def load_districts():

    print("\nLoading Bangladesh district boundaries...")

    with open(
        BOUNDARY_FILE,
        encoding="utf-8"
    ) as file:

        geojson = json.load(file)

    districts = []

    for feature in geojson["features"]:

        properties = feature["properties"]

        geometry = feature["geometry"]

        district = {
            "district_code": properties["adm2_pcode"],
            "district": properties["adm2_en"],
            "division": properties["adm1_en"],
            "geometry": geometry
        }

        districts.append(district)

    districts.sort(
        key=lambda x: x["district_code"]
    )

    print("Districts loaded:", len(districts))

    if len(districts) != 64:

        raise ValueError(
            f"Expected 64 districts, found {len(districts)}"
        )

    return districts


# ============================================================
# CHECK CACHED DISTRICT RESULT
# ============================================================

def load_cached_result(district_code):

    cache_file = RAW_DIR / f"{district_code}.json"

    if not cache_file.exists():
        return None

    try:

        with open(
            cache_file,
            encoding="utf-8"
        ) as file:

            cached = json.load(file)

        result = cached.get("result", {})

        pyramid = result.get("agesex_pyramid")

        if (
            cached.get("status") == "success"
            and isinstance(pyramid, list)
            and len(pyramid) > 0
        ):

            print(
                f"Using cached population for {district_code}"
            )

            return cached

    except (
        OSError,
        json.JSONDecodeError,
        AttributeError
    ):

        pass

    print(
        f"Invalid cache for {district_code}; downloading again."
    )

    return None


# ============================================================
# SUBMIT WORLDPOP AGE-SEX REQUEST
# ============================================================

def submit_population_request(district):

    print(
        f"\nSubmitting WorldPop request: "
        f"{district['district']} ({district['district_code']})"
    )

    geometry = prepare_geometry(
        district["geometry"],
        district["district"]
    )

    payload = {
        "geojson": geometry,
        "year": YEAR,
        "resolution": RESOLUTION,
        "sex": SEX
    }

    response = request_with_retry(
        "POST",
        f"{API_URL}/agesex",
        json=payload
    )

    task_id = response.get("task_id")

    if not task_id:

        raise RuntimeError(
            "WorldPop did not return a task ID:\n"
            + json.dumps(response, indent=2)
        )

    print("Task ID:", task_id)

    return task_id


# ============================================================
# WAIT FOR WORLDPOP TASK
# ============================================================

def wait_for_result(task_id):

    task_url = f"{API_URL}/tasks/{task_id}"

    print("Waiting for population result...")

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):

        response = request_with_retry(
            "GET",
            task_url
        )

        status = response.get("status")

        print(f"Attempt {attempt}: {status}")

        if status == "success":

            result = response.get("result", {})

            if not result.get("agesex_pyramid"):

                raise ValueError(
                    "WorldPop returned no age-sex population data."
                )

            return response

        if status == "failure":

            raise RuntimeError(
                "WorldPop processing failed:\n"
                + json.dumps(response, indent=2)
            )

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"WorldPop task {task_id} did not finish in time."
    )


# ============================================================
# DOWNLOAD ONE DISTRICT
# ============================================================

def download_district(district):

    district_code = district["district_code"]

    # Reuse previous successful downloads.
    cached = load_cached_result(district_code)

    if cached is not None:
        return cached

    task_id = submit_population_request(district)

    result = wait_for_result(task_id)

    # Save original API response.
    cache_file = RAW_DIR / f"{district_code}.json"

    with open(
        cache_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print(
        f"Population downloaded: {district['district']}"
    )

    return result


# ============================================================
# CONVERT ONE DISTRICT RESULT INTO CSV ROWS
# ============================================================

def convert_district_result(district, response):

    result = response["result"]

    pyramid = result["agesex_pyramid"]

    rows = []

    for age_group in pyramid:

        male = float(age_group["male"])
        female = float(age_group["female"])

        rows.append({

            "district_code": district["district_code"],

            "district": district["district"],

            "division": district["division"],

            "year": result.get("data_year", YEAR),

            "age_class": age_group["class"],

            "age_group": age_group["age"],

            "male_population": male,

            "female_population": female,

            "total_population": male + female,

            "resolution": result.get(
                "resolution",
                RESOLUTION
            ),

            "area_km2": result.get("area_km2"),

            "data_source": result.get(
                "data_source",
                "WorldPop"
            )

        })

    district_total = {

        "district_code": district["district_code"],

        "district": district["district"],

        "division": district["division"],

        "year": result.get("data_year", YEAR),

        "total_population": result.get(
            "total_population"
        ),

        "area_km2": result.get("area_km2"),

        "data_source": result.get(
            "data_source",
            "WorldPop"
        )

    }

    return rows, district_total


# ============================================================
# SAVE COMBINED POPULATION FILES
# ============================================================

def save_outputs(all_rows, all_totals):

    population_columns = [

        "district_code",
        "district",
        "division",
        "year",
        "age_class",
        "age_group",
        "male_population",
        "female_population",
        "total_population",
        "resolution",
        "area_km2",
        "data_source"

    ]

    totals_columns = [

        "district_code",
        "district",
        "division",
        "year",
        "total_population",
        "area_km2",
        "data_source"

    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=population_columns
        )

        writer.writeheader()
        writer.writerows(all_rows)

    with open(
        TOTALS_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=totals_columns
        )

        writer.writeheader()
        writer.writerows(all_totals)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EPISCOPE - BANGLADESH POPULATION DOWNLOAD")
    print("Source: WorldPop")
    print("Year:", YEAR)
    print("Resolution:", RESOLUTION)
    print("=" * 60)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    districts = load_districts()

    all_rows = []
    all_totals = []

    completed_count = 0

    for index, district in enumerate(districts, start=1):

        print("\n" + "-" * 55)
        print(f"DISTRICT {index}/{len(districts)}")

        response = download_district(district)

        rows, district_total = convert_district_result(
            district,
            response
        )

        all_rows.extend(rows)
        all_totals.append(district_total)

        # Save after every district.
        save_outputs(
            all_rows,
            all_totals
        )

        completed_count += 1

        print("Age groups collected:", len(rows))

        print(
            "District population:",
            district_total["total_population"]
        )

        print(
            "Completed districts:",
            completed_count,
            "/",
            len(districts)
        )

    print("\n" + "=" * 60)
    print("WORLDPOP DOWNLOAD COMPLETE!")
    print("=" * 60)

    print("Districts:", len(all_totals))
    print("Age-sex records:", len(all_rows))
    print("Population year:", YEAR)
    print("Resolution:", RESOLUTION)
    print("Age-sex dataset:", OUTPUT_FILE)
    print("District totals:", TOTALS_FILE)
    print("Raw responses:", RAW_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
