
import json
import time
import requests

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "bangladesh_districts.geojson"
)

API_URL = "https://api.worldpop.org/v2"

YEAR = 2022
RESOLUTION = "1km"


def main():

    print("=" * 55)
    print("EPISCOPE - WORLDPOP API TEST")
    print("=" * 55)

    # Load district boundaries
    with open(BOUNDARY_FILE, encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson["features"]

    print("District features:", len(features))

    # Test with the first district
    feature = features[0]

    properties = feature["properties"]

    district_name = properties["adm2_en"]
    district_code = properties["adm2_pcode"]

    geometry = feature["geometry"]

    print("Testing district:", district_name)
    print("District code:", district_code)
    print("Geometry type:", geometry["type"])

    # Submit WorldPop age-sex request
    payload = {
        "geojson": geometry,
        "year": YEAR,
        "resolution": RESOLUTION,
        "sex": "both"
    }

    print("\nSubmitting request to WorldPop...")

    response = requests.post(
        f"{API_URL}/agesex",
        json=payload,
        timeout=120
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    print("\nSubmission response:")
    print(json.dumps(result, indent=2)[:3000])

    task_id = result.get("task_id")

    if not task_id:
        raise RuntimeError(
            "WorldPop did not return a task_id. "
            "Check the response above."
        )

    print("\nTask ID:", task_id)

    # Poll until the task completes
    print("\nWaiting for WorldPop result...")

    for attempt in range(60):

        status_response = requests.get(
            f"{API_URL}/tasks/{task_id}",
            timeout=60
        )

        status_response.raise_for_status()

        status_data = status_response.json()

        status = status_data.get("status")

        print(
            f"Attempt {attempt + 1}: {status}"
        )

        if status == "success":

            print("\nWORLDPOP REQUEST SUCCESSFUL!")

            print("\nResult:")
            print(
                json.dumps(
                    status_data.get("result", {}),
                    indent=2
                )[:10000]
            )

            output_file = (
                PROJECT_ROOT
                / "data"
                / "reference"
                / "worldpop_test_result.json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    status_data,
                    file,
                    indent=2
                )

            print("\nSaved test response:")
            print(output_file)

            return

        if status == "failure":

            raise RuntimeError(
                "WorldPop task failed:\n"
                + json.dumps(status_data, indent=2)
            )

        time.sleep(5)

    raise TimeoutError(
        "WorldPop task did not finish within 5 minutes."
    )


if __name__ == "__main__":
    main()
