
import os
import glob
import json
import pandas as pd
import pulp

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# ============================================================
# EPISCOPE: RESOURCE ALLOCATION AND OPTIMISATION
# Planning period: Forecast horizon 1
# Forecast week: 2025-12-29
# Stock snapshot: 2025-12-31
#
# IMPORTANT:
# Resource-use coefficients are illustrative simulation
# assumptions, NOT clinical guidelines.
# ============================================================

FORECAST_PATH = "data/gold/surge_aware_forecasts"
FACILITY_PATH = "data/bronze/facilities/facilities.csv"
STOCK_PATTERN = "data/bronze/stock/year=2025/month=12/*.jsonl"

OUTPUT_DIR = "data/gold/resource_optimisation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_HORIZON = 1
EXPECTED_STOCK_DATE = "2025-12-31"


# ============================================================
# 1. ILLUSTRATIVE RESOURCE REQUIREMENTS
# ============================================================
# Units are hypothetical planning units per predicted case.
# These assumptions exist only to demonstrate the LP model.

RESOURCE_COEFFICIENTS = {
    "Dengue": {
        "iv_saline": 0.25,
        "ns1_kits": 0.50,
        "beds": 0.08
    },
    "Diarrhoea": {
        "iv_saline": 0.10,
        "ors": 2.00,
        "beds": 0.03
    },
    "Influenza": {
        "iv_saline": 0.05,
        "beds": 0.02
    },
    "Malaria": {
        "antimalarial": 1.00,
        "beds": 0.05
    }
}

STOCK_COLUMNS = {
    "iv_saline": "iv_saline_stock",
    "ns1_kits": "ns1_kits_stock",
    "ors": "ors_stock",
    "antimalarial": "antimalarial_stock"
}

# Relative priorities used by the optimisation objective.
# These are modelling weights, not clinical priority rankings.
RESOURCE_PRIORITIES = {
    "iv_saline": 1.0,
    "ns1_kits": 1.0,
    "ors": 1.0,
    "antimalarial": 1.0,
    "beds": 1.5
}


def district_key(value):
    return str(value).strip().lower()


# ============================================================
# 2. LOAD FORECAST DATA
# ============================================================

print("\n[1/7] Loading forecast data...")

spark = (
    SparkSession.builder
    .appName("EpiScopeResourceOptimisation")
    .getOrCreate()
)

forecast_spark = spark.read.parquet(FORECAST_PATH)

forecast = (
    forecast_spark
    .filter(col("forecast_horizon") == TARGET_HORIZON)
    .select(
        "district",
        "district_code",
        "origin_week",
        "forecast_week",
        "forecast_horizon",
        "disease",
        "predicted_cases"
    )
    .toPandas()
)

spark.stop()

if forecast.empty:
    raise ValueError("No forecast records found for horizon 1.")

forecast["predicted_cases"] = (
    pd.to_numeric(forecast["predicted_cases"], errors="coerce")
    .fillna(0)
    .clip(lower=0)
)

forecast["district_key"] = forecast["district"].apply(district_key)

print("Forecast records:", len(forecast))
print("Forecast week:", forecast["forecast_week"].min())
print("Forecast diseases:", sorted(forecast["disease"].unique()))


# ============================================================
# 3. LOAD FACILITIES
# ============================================================

print("\n[2/7] Loading synthetic facilities...")

facilities = pd.read_csv(FACILITY_PATH)

required_facility_columns = [
    "facility_id",
    "district",
    "beds"
]

for column in required_facility_columns:
    if column not in facilities.columns:
        raise ValueError(f"Missing facility column: {column}")

facilities["facility_id"] = facilities["facility_id"].astype(str)
facilities["district_key"] = facilities["district"].apply(district_key)

facilities["beds"] = (
    pd.to_numeric(facilities["beds"], errors="coerce")
    .fillna(0)
    .clip(lower=0)
)

print("Facilities:", len(facilities))
print("Districts:", facilities["district_key"].nunique())


# ============================================================
# 4. LOAD LATEST STOCK SNAPSHOT
# ============================================================

print("\n[3/7] Loading stock records...")

stock_files = glob.glob(STOCK_PATTERN)

if not stock_files:
    raise FileNotFoundError("December 2025 stock files not found.")

stock_frames = [
    pd.read_json(path, lines=True)
    for path in stock_files
]

stock = pd.concat(stock_frames, ignore_index=True)

stock["facility_id"] = stock["facility_id"].astype(str)
stock["snapshot_date"] = pd.to_datetime(
    stock["snapshot_date"],
    errors="coerce"
)

latest_stock_date = stock["snapshot_date"].max()

if pd.isna(latest_stock_date):
    raise ValueError("Could not determine latest stock date.")

if latest_stock_date.strftime("%Y-%m-%d") != EXPECTED_STOCK_DATE:
    raise ValueError(
        f"Expected stock date {EXPECTED_STOCK_DATE}, "
        f"found {latest_stock_date.date()}"
    )

# Select one latest record per facility.
stock = (
    stock.sort_values("snapshot_date")
    .drop_duplicates(subset=["facility_id"], keep="last")
)

if stock["facility_id"].nunique() != len(facilities):
    raise ValueError(
        "Stock records do not cover all facilities. "
        "Check facility IDs before optimisation."
    )

print("Latest stock date:", latest_stock_date.date())
print("Facilities with stock:", stock["facility_id"].nunique())


# ============================================================
# 5. PREPARE FACILITY RESOURCE AVAILABILITY
# ============================================================

print("\n[4/7] Preparing resource availability...")

stock_columns = list(STOCK_COLUMNS.values()) + ["occupied_beds"]

for column in stock_columns:
    if column not in stock.columns:
        raise ValueError(f"Missing stock column: {column}")

for column in stock_columns:
    stock[column] = (
        pd.to_numeric(stock[column], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

facilities = facilities.merge(
    stock[["facility_id"] + stock_columns],
    on="facility_id",
    how="left",
    validate="one_to_one"
)

for column in stock_columns:
    facilities[column] = facilities[column].fillna(0)

facilities["available_beds"] = (
    facilities["beds"] - facilities["occupied_beds"]
).clip(lower=0)

facilities["district_key"] = facilities["district"].apply(district_key)

# Aggregate stock by district. The LP treats stock as transferable
# between facilities within the same district.
district_stock = (
    facilities.groupby("district_key")[stock_columns]
    .sum()
    .to_dict("index")
)

print("Facilities prepared:", len(facilities))
print(
    "Total available beds:",
    round(facilities["available_beds"].sum(), 2)
)


# ============================================================
# 6. ESTIMATE FACILITY-LEVEL RESOURCE DEMAND
# ============================================================

print("\n[5/7] Estimating resource demand...")

demand_rows = []

for _, forecast_row in forecast.iterrows():

    district = forecast_row["district"]
    key = forecast_row["district_key"]
    disease = forecast_row["disease"]
    cases = float(forecast_row["predicted_cases"])

    if disease not in RESOURCE_COEFFICIENTS:
        continue

    district_facilities = facilities[
        facilities["district_key"] == key
    ].copy()

    if district_facilities.empty:
        print("WARNING: No facilities for district:", district)
        continue

    # Distribute predicted cases by facility bed capacity.
    # This is a simulation allocation assumption.
    total_beds = district_facilities["beds"].sum()

    if total_beds > 0:
        district_facilities["case_share"] = (
            district_facilities["beds"] / total_beds
        )
    else:
        district_facilities["case_share"] = (
            1.0 / len(district_facilities)
        )

    for _, facility in district_facilities.iterrows():

        facility_cases = cases * float(facility["case_share"])

        for resource, coefficient in RESOURCE_COEFFICIENTS[disease].items():

            required_quantity = facility_cases * coefficient

            demand_rows.append({
                "district": district,
                "district_key": key,
                "facility_id": facility["facility_id"],
                "facility_name": facility.get("facility_name", ""),
                "disease": disease,
                "resource": resource,
                "predicted_cases_at_facility": facility_cases,
                "demand": required_quantity
            })

demand = pd.DataFrame(demand_rows)

if demand.empty:
    raise ValueError("No resource demand could be generated.")

print("Facility-disease-resource demand rows:", len(demand))


# ============================================================
# 7. BUILD LINEAR PROGRAMMING MODEL
# ============================================================

print("\n[6/7] Building PuLP optimisation model...")

model = pulp.LpProblem(
    "EpiScope_Resource_Allocation",
    pulp.LpMaximize
)

allocation_variables = {}

for index, row in demand.iterrows():

    allocation_variables[index] = pulp.LpVariable(
        f"allocation_{index}",
        lowBound=0,
        upBound=float(row["demand"]),
        cat="Continuous"
    )


# Objective:
# Maximise weighted demand fulfilment.
# Each allocation is divided by its demand so the objective
# rewards fulfilment rates rather than simply large quantities.

model += pulp.lpSum(
    RESOURCE_PRIORITIES[row["resource"]]
    * allocation_variables[index]
    / max(float(row["demand"]), 1.0)
    for index, row in demand.iterrows()
)


# ------------------------------------------------------------
# A. District-level stock constraints
# ------------------------------------------------------------

for district in demand["district_key"].unique():

    district_rows = demand[
        demand["district_key"] == district
    ]

    for resource, stock_column in STOCK_COLUMNS.items():

        relevant_indices = district_rows[
            district_rows["resource"] == resource
        ].index.tolist()

        if not relevant_indices:
            continue

        available = float(
            district_stock.get(district, {}).get(stock_column, 0)
        )

        model += (
            pulp.lpSum(
                allocation_variables[i]
                for i in relevant_indices
            ) <= available,
            f"Stock_{district}_{resource}"
        )


# ------------------------------------------------------------
# B. Facility-level available bed constraints
# ------------------------------------------------------------

bed_rows = demand[demand["resource"] == "beds"]

for facility_id in facilities["facility_id"]:

    relevant_indices = bed_rows[
        bed_rows["facility_id"] == facility_id
    ].index.tolist()

    if not relevant_indices:
        continue

    facility_record = facilities[
        facilities["facility_id"] == facility_id
    ].iloc[0]

    available_beds = float(facility_record["available_beds"])

    model += (
        pulp.lpSum(
            allocation_variables[i]
            for i in relevant_indices
        ) <= available_beds,
        f"BedCapacity_{facility_id}"
    )


# ============================================================
# 8. SOLVE MODEL
# ============================================================

print("Solving optimisation model...")

solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

status = pulp.LpStatus[model.status]

print("\nOptimisation status:", status)

if status != "Optimal":
    raise RuntimeError(
        f"Optimisation did not reach an optimal solution: {status}"
    )


# ============================================================
# 9. BUILD FACILITY-LEVEL RESULTS
# ============================================================

print("\n[7/7] Preparing and saving results...")

allocation_results = []

for index, row in demand.iterrows():

    allocated = float(
        pulp.value(allocation_variables[index]) or 0
    )

    required = float(row["demand"])
    unmet = max(required - allocated, 0)

    allocation_results.append({
        **row.to_dict(),
        "allocated_quantity": allocated,
        "unmet_quantity": unmet,
        "fulfilment_rate": (
            allocated / required if required > 0 else 1.0
        )
    })

allocation_df = pd.DataFrame(allocation_results)

allocation_path = os.path.join(
    OUTPUT_DIR,
    "facility_resource_allocation.csv"
)

allocation_df.to_csv(allocation_path, index=False)


# ============================================================
# 10. DISTRICT RESOURCE SUMMARY
# ============================================================

summary = (
    allocation_df.groupby(
        ["district", "district_key", "resource"],
        as_index=False
    )
    .agg(
        total_demand=("demand", "sum"),
        total_allocated=("allocated_quantity", "sum"),
        total_unmet=("unmet_quantity", "sum")
    )
)

available_values = []

for _, row in summary.iterrows():

    if row["resource"] == "beds":
        available = float(
            facilities.loc[
                facilities["district_key"] == row["district_key"],
                "available_beds"
            ].sum()
        )
    else:
        stock_column = STOCK_COLUMNS[row["resource"]]
        available = float(
            district_stock.get(row["district_key"], {}).get(
                stock_column, 0
            )
        )

    available_values.append(available)

summary["available_stock_or_capacity"] = available_values

summary["fulfilment_rate"] = (
    summary["total_allocated"]
    / summary["total_demand"].replace(0, pd.NA)
).fillna(1.0)

summary_path = os.path.join(
    OUTPUT_DIR,
    "district_resource_summary.csv"
)

summary.to_csv(summary_path, index=False)


# ============================================================
# 11. SAVE MODEL METADATA
# ============================================================

metadata = {
    "project": "EpiScope",
    "module": "Resource Allocation and Optimisation",
    "optimisation_status": status,
    "forecast_horizon": TARGET_HORIZON,
    "forecast_origin": str(forecast["origin_week"].min()),
    "forecast_week": str(forecast["forecast_week"].min()),
    "stock_snapshot_date": EXPECTED_STOCK_DATE,
    "facility_count": int(len(facilities)),
    "district_count": int(facilities["district_key"].nunique()),
    "demand_rows": int(len(demand)),
    "resource_coefficients": RESOURCE_COEFFICIENTS,
    "assumptions": [
        "Facilities and stock records are synthetic.",
        "Resource-use coefficients are illustrative simulation assumptions.",
        "Predicted cases are distributed across facilities by bed capacity.",
        "Stock is assumed transferable between facilities within each district.",
        "Available beds equal beds minus occupied beds.",
        "This is not a clinical or real-world procurement recommendation."
    ]
}

metadata_path = os.path.join(
    OUTPUT_DIR,
    "optimisation_metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# 12. VALIDATION AND SUMMARY
# ============================================================

print("\n========== OPTIMISATION SUMMARY ==========")

print("Status:", status)
print("Forecast week:", forecast["forecast_week"].min())
print("Stock snapshot:", EXPECTED_STOCK_DATE)
print("Facilities:", len(facilities))
print("Demand rows:", len(demand))
print("Allocation rows:", len(allocation_df))

print(
    "Total simulated demand:",
    round(allocation_df["demand"].sum(), 2)
)

print(
    "Total allocated:",
    round(allocation_df["allocated_quantity"].sum(), 2)
)

print(
    "Total unmet demand:",
    round(allocation_df["unmet_quantity"].sum(), 2)
)

overall_demand = allocation_df["demand"].sum()
overall_allocated = allocation_df["allocated_quantity"].sum()

overall_fulfilment = (
    overall_allocated / overall_demand * 100
    if overall_demand > 0 else 100
)

print(
    "Overall weighted-quantity fulfilment:",
    round(overall_fulfilment, 2),
    "%"
)

print("\nResource summary:")
print(
    summary.groupby("resource")[
        ["total_demand", "total_allocated", "total_unmet"]
    ].sum().round(2).to_string()
)

print("\nLowest fulfilment districts:")
print(
    summary.sort_values("fulfilment_rate")
    [["district", "resource", "fulfilment_rate", "total_unmet"]]
    .head(10)
    .to_string(index=False)
)

# Basic validation
assert allocation_df["allocated_quantity"].ge(-0.000001).all()
assert allocation_df["unmet_quantity"].ge(-0.000001).all()
assert (
    allocation_df["allocated_quantity"]
    <= allocation_df["demand"] + 0.0001
).all()

print("\nSaved files:")
print(allocation_path)
print(summary_path)
print(metadata_path)

print("\nEPISCOPE RESOURCE OPTIMISATION COMPLETED!")
