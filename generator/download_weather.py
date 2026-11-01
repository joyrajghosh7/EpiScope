
import csv
import json
import time
import requests

from pathlib import Path
from collections import defaultdict


# ============================================================
# EPISCOPE - HISTORICAL WEATHER DOWNLOADER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACILITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "facilities"
    / "facilities.csv"
)

BRONZE_DIR = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "weather"
)

RAW_DIR = BRONZE_DIR / "raw"

OUTPUT_FILE = BRONZE_DIR / "weather_daily.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

BATCH_SIZE = 8
BATCH_DELAY = 30
MAX_RETRIES = 5

# IMPORTANT: Use ERA5, not ERA5-Land.
# ERA5 provides precipitation, temperature and humidity.

DAILY_VARIABLES = (
    "precipitation_sum,"
    "temperature_2m_mean,"
    "relative_humidity_2m_mean"
)

WEATHER_SOURCE = "Open-Meteo ERA5"


# ============================================================
# LOAD DISTRICT LOCATIONS
# ============================================================

def load_district_locations():

    print("\nLoading district locations...")

    grouped = defaultdict(list)

    with open(
        FACILITY_FILE,
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            grouped[row["district_code"]].append(row)

    locations = []

    for district_code, facilities in sorted(grouped.items()):

        latitude = sum(
            float(f["latitude"])
            for f in facilities
        ) / len(facilities)

        longitude = sum(
            float(f["longitude"])
            for f in facilities
        ) / len(facilities)

        locations.append({

            "district_code": district_code,

            "district": facilities[0]["district"],

            "latitude": round(latitude, 5),

            "longitude": round(longitude, 5)

        })

    print("Districts found:", len(locations))

    if len(locations) != 64:

        raise ValueError(
            f"Expected 64 districts, found {len(locations)}"
        )

    return locations


# ============================================================
# DOWNLOAD ONE BATCH
# ============================================================

def download_batch(batch, batch_number):

    raw_file = (
        RAW_DIR
        / f"weather_batch_{batch_number:02d}.json"
    )

    # --------------------------------------------------------
    # REUSE CACHE ONLY IF IT HAS ALL REQUIRED DATA
    # --------------------------------------------------------

    if raw_file.exists():

        print(
            f"\nChecking cached batch {batch_number}..."
        )

        with open(
            raw_file,
            encoding="utf-8"
        ) as file:

            cached_data = json.load(file)

        cache_is_valid = (
            isinstance(cached_data, list)
            and len(cached_data) == len(batch)
        )

        if cache_is_valid:

            for weather in cached_data:

                daily = weather.get("daily", {})

                required = [
                    "time",
                    "precipitation_sum",
                    "temperature_2m_mean",
                    "relative_humidity_2m_mean"
                ]

                if any(
                    field not in daily
                    for field in required
                ):
                    cache_is_valid = False
                    break

                # Reject the old ERA5-Land cache if
                # precipitation is missing or entirely null.
                if not any(
                    value is not None
                    for value in daily["precipitation_sum"]
                ):
                    cache_is_valid = False
                    break

            if cache_is_valid:

                print(
                    f"Using valid cached batch {batch_number}..."
                )

                return cached_data

        print(
            "Old or invalid cache found. Downloading again..."
        )

        raw_file.unlink(missing_ok=True)

    # --------------------------------------------------------
    # API PARAMETERS
    # --------------------------------------------------------

    params = {

        "latitude": ",".join(
            str(x["latitude"])
            for x in batch
        ),

        "longitude": ",".join(
            str(x["longitude"])
            for x in batch
        ),

        "start_date": START_DATE,

        "end_date": END_DATE,

        "daily": DAILY_VARIABLES,

        "timezone": "Asia/Dhaka",

        "models": "era5",

        "temperature_unit": "celsius",

        "precipitation_unit": "mm"

    }

    print(
        f"\nDownloading ERA5 weather batch {batch_number}..."
    )

    # --------------------------------------------------------
    # AUTOMATIC RETRY
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=180
            )

            # Handle API rate limiting.
            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                try:
                    wait_seconds = int(retry_after)
                except (TypeError, ValueError):
                    wait_seconds = 60 * attempt

                wait_seconds = max(
                    wait_seconds,
                    60 * attempt
                )

                print("\nAPI RATE LIMIT (429)")

                print(
                    f"Attempt {attempt}/{MAX_RETRIES}"
                )

                print(
                    f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

                continue

            response.raise_for_status()

            result = response.json()

            # Normalize one location to a list.
            if isinstance(result, dict):
                result = [result]

            if len(result) != len(batch):

                raise ValueError(
                    f"Expected {len(batch)} locations, "
                    f"received {len(result)}"
                )

            # ------------------------------------------------
            # VALIDATE API RESPONSE BEFORE CACHING
            # ------------------------------------------------

            for weather in result:

                if "daily" not in weather:

                    raise ValueError(
                        "Daily weather data is missing"
                    )

                daily = weather["daily"]

                required = [
                    "time",
                    "precipitation_sum",
                    "temperature_2m_mean",
                    "relative_humidity_2m_mean"
                ]

                for field in required:

                    if field not in daily:

                        raise ValueError(
                            f"Missing API field: {field}"
                        )

                    if len(daily[field]) != len(daily["time"]):

                        raise ValueError(
                            f"Inconsistent data length: {field}"
                        )

                # Precipitation must contain actual values.
                if not any(
                    value is not None
                    for value in daily["precipitation_sum"]
                ):

                    raise ValueError(
                        "ERA5 returned no precipitation values"
                    )

            # Save only a validated response.
            with open(
                raw_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2
                )

            print(
                f"Batch {batch_number} downloaded successfully."
            )

            return result

        except requests.RequestException as error:

            print(
                f"\nRequest error: {error}"
            )

            if attempt == MAX_RETRIES:
                raise

            wait_seconds = 30 * attempt

            print(
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to download batch {batch_number}"
    )


# ============================================================
# CONVERT WEATHER DATA INTO CSV RECORDS
# ============================================================

def convert_to_records(batch, weather_results):

    records = []

    for location, weather in zip(
        batch,
        weather_results
    ):

        daily = weather["daily"]

        dates = daily["time"]

        for i, weather_date in enumerate(dates):

            records.append({

                "district_code": location["district_code"],

                "district": location["district"],

                "date": weather_date,

                "latitude": location["latitude"],

                "longitude": location["longitude"],

                "precipitation_mm": (
                    daily["precipitation_sum"][i]
                ),

                "temperature_mean_c": (
                    daily["temperature_2m_mean"][i]
                ),

                "humidity_mean_percent": (
                    daily["relative_humidity_2m_mean"][i]
                ),

                "weather_source": WEATHER_SOURCE

            })

    return records


# ============================================================
# MAIN DOWNLOAD PROCESS
# ============================================================

def main():

    print("=" * 55)

    print("EPISCOPE - HISTORICAL WEATHER DOWNLOAD")

    print("Weather model: ERA5")

    print("=" * 55)

    BRONZE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    locations = load_district_locations()

    all_records = []

    total_batches = (
        len(locations) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    # --------------------------------------------------------
    # DOWNLOAD ALL DISTRICTS
    # --------------------------------------------------------

    for start in range(
        0,
        len(locations),
        BATCH_SIZE
    ):

        batch = locations[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        print("\n" + "-" * 45)

        print(
            f"Batch {batch_number}/{total_batches}"
        )

        weather_results = download_batch(
            batch,
            batch_number
        )

        batch_records = convert_to_records(
            batch,
            weather_results
        )

        all_records.extend(batch_records)

        print(
            "Districts completed:",
            min(start + BATCH_SIZE, len(locations)),
            "/",
            len(locations)
        )

        print(
            "Records collected:",
            len(all_records)
        )

        if batch_number < total_batches:

            print(
                f"Waiting {BATCH_DELAY} seconds..."
            )

            time.sleep(BATCH_DELAY)

    # --------------------------------------------------------
    # SAVE COMBINED DATASET
    # --------------------------------------------------------

    columns = [

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

    print("\nSaving combined weather dataset...")

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(all_records)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 55)

    print("WEATHER DOWNLOAD COMPLETE!")

    print("=" * 55)

    print("Weather model: ERA5")

    print("Districts:", len(locations))

    print("Weather records:", len(all_records))

    print("Expected records:", 64 * 1826)

    print("Raw API responses:", RAW_DIR)

    print("Combined CSV:", OUTPUT_FILE)

    print("=" * 55)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
