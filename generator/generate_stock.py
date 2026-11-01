
import json
import math
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import date, timedelta


# ============================================================
# EPISCOPE - SYNTHETIC FACILITY STOCK GENERATOR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACILITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "facilities"
    / "facilities.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "stock"
)

START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)

SEED = 42

STOCK_NAMES = [
    "iv_saline_stock",
    "ns1_kits_stock",
    "ors_stock",
    "antimalarial_stock"
]


# ============================================================
# FACILITY-SPECIFIC INVENTORY CAPACITY
# ============================================================

def get_inventory_capacity(beds):

    return {
        "iv_saline_stock": 30 + beds * 5,
        "ns1_kits_stock": 10 + beds * 2,
        "ors_stock": 30 + beds * 4,
        "antimalarial_stock": 10 + beds * 1.5
    }


# ============================================================
# FACILITY OCCUPANCY TARGET
# ============================================================

def get_occupancy_rate(facility_type, month):

    rates = {
        "Hospital": 0.72,
        "Upazila Health Complex": 0.55,
        "Clinic": 0.35
    }

    rate = rates.get(facility_type, 0.50)

    # Slightly higher seasonal healthcare demand.
    if month in [6, 7, 8, 9, 10]:
        rate += 0.05

    return min(rate, 0.90)


# ============================================================
# INITIALIZE ONE FACILITY
# ============================================================

def initialize_facility(facility, rng):

    beds = int(facility["beds"])

    capacity = get_inventory_capacity(beds)

    stocks = {}

    for name in STOCK_NAMES:

        # Initial stock varies between 50% and 100% capacity.
        stocks[name] = int(
            rng.uniform(0.50, 1.00) * capacity[name]
        )

    initial_occupancy = int(
        round(beds * rng.uniform(0.35, 0.75))
    )

    return {
        "stocks": stocks,
        "capacity": capacity,
        "occupied_beds": min(initial_occupancy, beds),
        "pending_deliveries": {
            name: None for name in STOCK_NAMES
        }
    }


# ============================================================
# APPLY DELIVERIES AND CONSUMPTION
# ============================================================

def update_stock(state, stock_name, daily_demand, day_number, rng):

    stocks = state["stocks"]
    capacity = state["capacity"]
    pending = state["pending_deliveries"]

    # Apply delivery if its scheduled date has arrived.
    delivery = pending[stock_name]

    if delivery is not None:

        delivery_day, quantity = delivery

        if day_number >= delivery_day:

            stocks[stock_name] = min(
                capacity[stock_name],
                stocks[stock_name] + quantity
            )

            pending[stock_name] = None

    # Consume supplies.
    actual_consumption = min(
        stocks[stock_name],
        daily_demand
    )

    stocks[stock_name] -= actual_consumption

    # Reorder when stock reaches a low level.
    reorder_level = 0.25 * capacity[stock_name]

    if (
        stocks[stock_name] <= reorder_level
        and pending[stock_name] is None
    ):

        # Variable delivery lead time.
        lead_time = int(rng.integers(2, 8))

        # Occasional supply-chain delay.
        if rng.random() < 0.10:
            lead_time += int(rng.integers(3, 9))

        target_stock = int(
            0.90 * capacity[stock_name]
        )

        order_quantity = max(
            0,
            target_stock - stocks[stock_name]
        )

        pending[stock_name] = (
            day_number + lead_time,
            order_quantity
        )

    return int(stocks[stock_name])


# ============================================================
# UPDATE OCCUPIED BEDS
# ============================================================

def update_occupancy(state, facility, current_date, rng):

    beds = int(facility["beds"])

    previous = state["occupied_beds"]

    target_rate = get_occupancy_rate(
        facility["facility_type"],
        current_date.month
    )

    target_occupancy = beds * target_rate

    # Gradually move toward target occupancy.
    movement = 0.15 * (
        target_occupancy - previous
    )

    # Random daily admissions/discharges variation.
    variation = rng.normal(
        0,
        max(1, beds * 0.025)
    )

    new_occupancy = round(
        previous + movement + variation
    )

    new_occupancy = max(
        0,
        min(beds, new_occupancy)
    )

    state["occupied_beds"] = int(new_occupancy)

    return int(new_occupancy)


# ============================================================
# GENERATE ONE DAILY FACILITY SNAPSHOT
# ============================================================

def generate_snapshot(facility, state, current_date, day_number, rng):

    occupied = update_occupancy(
        state,
        facility,
        current_date,
        rng
    )

    beds = int(facility["beds"])

    # Demand depends partly on facility activity.
    activity = max(0.1, occupied)

    demands = {

        "iv_saline_stock": int(
            rng.poisson(max(0.2, activity * 0.12))
        ),

        "ns1_kits_stock": int(
            rng.poisson(max(0.1, activity * 0.035))
        ),

        "ors_stock": int(
            rng.poisson(max(0.2, activity * 0.08))
        ),

        "antimalarial_stock": int(
            rng.poisson(max(0.1, activity * 0.025))
        )

    }

    record = {
        "snapshot_date": current_date.isoformat(),
        "facility_id": facility["facility_id"]
    }

    for stock_name in STOCK_NAMES:

        record[stock_name] = update_stock(
            state,
            stock_name,
            demands[stock_name],
            day_number,
            rng
        )

    record["occupied_beds"] = occupied

    return record


# ============================================================
# LOAD AND VALIDATE FACILITIES
# ============================================================

def load_facilities():

    print("\nLoading facility registry...")

    facilities = pd.read_csv(FACILITY_FILE)

    required_columns = [
        "facility_id",
        "district",
        "facility_type",
        "beds"
    ]

    missing = [
        column
        for column in required_columns
        if column not in facilities.columns
    ]

    if missing:
        raise ValueError(
            f"Missing facility columns: {missing}"
        )

    if facilities["facility_id"].duplicated().any():
        raise ValueError("Duplicate facility IDs found.")

    if facilities["beds"].isnull().any():
        raise ValueError("Missing bed capacity found.")

    if (facilities["beds"] <= 0).any():
        raise ValueError("Invalid bed capacity found.")

    print("Facilities loaded:", len(facilities))

    return facilities.to_dict("records")


# ============================================================
# MAIN GENERATOR
# ============================================================

def main():

    print("=" * 65)
    print("EPISCOPE - FACILITY STOCK AND BED OCCUPANCY GENERATOR")
    print("=" * 65)

    print("Start date:", START_DATE)
    print("End date:", END_DATE)
    print("Random seed:", SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    facilities = load_facilities()

    # Each facility has an independent random generator.
    states = []
    random_generators = []

    for index, facility in enumerate(facilities):

        rng = np.random.default_rng(SEED + index)

        state = initialize_facility(
            facility,
            rng
        )

        states.append(state)
        random_generators.append(rng)

    total_days = (
        END_DATE - START_DATE
    ).days + 1

    expected_records = total_days * len(facilities)

    print("\nExpected daily records:", expected_records)
    print("Facilities:", len(facilities))
    print("Days:", total_days)

    current_date = START_DATE
    day_number = 0

    current_month = None
    output_file = None

    total_records = 0
    total_stockouts = 0

    while current_date <= END_DATE:

        month_key = (
            current_date.year,
            current_date.month
        )

        # Create one JSONL file for each month.
        if month_key != current_month:

            if output_file is not None:
                output_file.close()

            year_dir = OUTPUT_DIR / f"year={current_date.year}"

            month_dir = (
                year_dir
                / f"month={current_date.month:02d}"
            )

            month_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                month_dir
                / f"stock_{current_date.year}_{current_date.month:02d}.jsonl"
            )

            output_file = open(
                output_path,
                "w",
                encoding="utf-8"
            )

            current_month = month_key

        daily_stockouts = 0

        for index, facility in enumerate(facilities):

            record = generate_snapshot(
                facility,
                states[index],
                current_date,
                day_number,
                random_generators[index]
            )

            output_file.write(
                json.dumps(record) + "\n"
            )

            total_records += 1

            for name in STOCK_NAMES:

                if record[name] == 0:
                    daily_stockouts += 1

        total_stockouts += daily_stockouts

        # Print progress at the end of each month.
        next_date = current_date + timedelta(days=1)

        if (
            next_date.month != current_date.month
            or next_date.year != current_date.year
            or current_date == END_DATE
        ):

            print(
                f"Completed {current_date.year}-{current_date.month:02d} | "
                f"Records: {total_records:,}"
            )

        current_date = next_date
        day_number += 1

    if output_file is not None:
        output_file.close()

    # Save generation metadata.
    summary = {
        "project": "EpiScope",
        "dataset": "Synthetic facility stock and bed occupancy",
        "source": "Synthetic simulation",
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "facilities": len(facilities),
        "days": total_days,
        "expected_records": expected_records,
        "generated_records": total_records,
        "random_seed": SEED,
        "stockout_observations": total_stockouts,
        "format": "JSON Lines",
        "partitioning": "year/month"
    }

    summary_file = (
        OUTPUT_DIR
        / "generation_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("STOCK GENERATION COMPLETE!")
    print("=" * 65)

    print("Facilities:", len(facilities))
    print("Days:", total_days)
    print("Generated records:", f"{total_records:,}")
    print("Expected records:", f"{expected_records:,}")
    print("Stockout observations:", f"{total_stockouts:,}")
    print("Output directory:", OUTPUT_DIR)
    print("Summary file:", summary_file)

    if total_records == expected_records:
        print("\nRECORD COUNT VALIDATION PASSED!")
    else:
        print("\nWARNING: RECORD COUNT MISMATCH!")


if __name__ == "__main__":
    main()
