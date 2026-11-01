
import json
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

FACILITY_FILE = (
    ROOT / "data" / "bronze" / "facilities" / "facilities.csv"
)

STOCK_DIR = ROOT / "data" / "bronze" / "stock"

STOCK_COLUMNS = [
    "iv_saline_stock",
    "ns1_kits_stock",
    "ors_stock",
    "antimalarial_stock"
]

EXPECTED_FACILITIES = 640
EXPECTED_DAYS = 1826
EXPECTED_RECORDS = EXPECTED_FACILITIES * EXPECTED_DAYS


def main():

    print("=" * 65)
    print("EPISCOPE - STOCK DATA VALIDATION")
    print("=" * 65)

    facilities = pd.read_csv(FACILITY_FILE)

    facility_ids = set(facilities["facility_id"])

    bed_capacity = dict(
        zip(facilities["facility_id"], facilities["beds"])
    )

    files = sorted(
        STOCK_DIR.glob("year=*/month=*/stock_*.jsonl")
    )

    errors = []

    total_records = 0
    total_stockouts = 0
    total_duplicate_records = 0

    all_dates = set()
    seen_facilities = set()

    print("\nMonthly files found:", len(files))

    if len(files) != 60:
        errors.append("Expected 60 monthly JSONL files.")

    # Validate each monthly file.
    for file in files:

        try:
            df = pd.read_json(file, lines=True)

        except Exception as error:
            errors.append(f"Could not read {file}: {error}")
            continue

        month_records = len(df)
        total_records += month_records

        required = [
            "snapshot_date",
            "facility_id",
            "occupied_beds"
        ] + STOCK_COLUMNS

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            errors.append(
                f"{file.name}: missing columns {missing}"
            )
            continue

        # Parse dates.
        dates = pd.to_datetime(
            df["snapshot_date"],
            errors="coerce"
        )

        if dates.isnull().any():
            errors.append(
                f"{file.name}: invalid dates found."
            )
            continue

        # Confirm records belong to the correct month.
        folder_year = int(file.parent.parent.name.split("=")[1])
        folder_month = int(file.parent.name.split("=")[1])

        if not (
            (dates.dt.year == folder_year)
            & (dates.dt.month == folder_month)
        ).all():

            errors.append(
                f"{file.name}: records outside partition month."
            )

        all_dates.update(
            dates.dt.strftime("%Y-%m-%d").unique()
        )

        # Facility ID validation.
        unknown_ids = set(df["facility_id"]) - facility_ids

        if unknown_ids:
            errors.append(
                f"{file.name}: unknown facility IDs found."
            )

        seen_facilities.update(df["facility_id"].unique())

        # Duplicate facility-date validation.
        duplicates = df.duplicated(
            subset=["snapshot_date", "facility_id"]
        ).sum()

        total_duplicate_records += int(duplicates)

        # Numeric validation.
        numeric_columns = STOCK_COLUMNS + ["occupied_beds"]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        if df[numeric_columns].isnull().any().any():
            errors.append(
                f"{file.name}: missing or invalid numeric values."
            )
            continue

        if (df[numeric_columns] < 0).any().any():
            errors.append(
                f"{file.name}: negative stock or occupancy values."
            )

        # Bed occupancy cannot exceed facility capacity.
        capacities = df["facility_id"].map(bed_capacity)

        invalid_beds = (
            df["occupied_beds"] > capacities
        ).sum()

        if invalid_beds:
            errors.append(
                f"{file.name}: {invalid_beds} occupancy values "
                "exceed bed capacity."
            )

        # Count stockout observations.
        total_stockouts += int(
            (df[STOCK_COLUMNS] == 0).sum().sum()
        )

        # Every date should have all 640 facilities.
        daily_counts = df.groupby("snapshot_date").size()

        if not (daily_counts == EXPECTED_FACILITIES).all():
            errors.append(
                f"{file.name}: some dates do not have "
                "exactly 640 facility records."
            )

        expected_month_records = (
            dates.dt.days_in_month.iloc[0]
            * EXPECTED_FACILITIES
        )

        if month_records != expected_month_records:
            errors.append(
                f"{file.name}: expected "
                f"{expected_month_records} records, "
                f"found {month_records}."
            )

        print(
            f"Validated {file.parent.parent.name}/"
            f"{file.parent.name}: "
            f"{month_records:,} records"
        )

    # Final summary.
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print("Total records:", f"{total_records:,}")
    print("Expected records:", f"{EXPECTED_RECORDS:,}")
    print("Unique facilities:", len(seen_facilities))
    print("Unique dates:", len(all_dates))
    print("Duplicate facility-date records:", total_duplicate_records)
    print("Stockout observations:", total_stockouts)

    if total_records != EXPECTED_RECORDS:
        errors.append("Total record count mismatch.")

    if len(seen_facilities) != EXPECTED_FACILITIES:
        errors.append("Expected 640 unique facilities.")

    if len(all_dates) != EXPECTED_DAYS:
        errors.append("Expected 1826 unique dates.")

    if total_duplicate_records > 0:
        errors.append("Duplicate facility-date records found.")

    print("\n" + "=" * 65)

    if errors:

        print("VALIDATION FAILED")

        for error in errors:
            print("ERROR:", error)

    else:

        print("STOCK DATA VALIDATION PASSED!")

    print("=" * 65)


if __name__ == "__main__":
    main()
