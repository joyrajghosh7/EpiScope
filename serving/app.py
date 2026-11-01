
from pathlib import Path
from functools import lru_cache
from typing import Optional
import csv
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dashboard" / "static" / "data"

DISTRICT_FILE = DATA_DIR / "district_summary.json"
ALERT_FILE = DATA_DIR / "outbreak_alerts_improved.json"
ORIGINAL_ALERT_FILE = DATA_DIR / "outbreak_alerts_original.json"
FORECAST_FILE = DATA_DIR / "forecasts.json"
WEEKLY_FILE = DATA_DIR / "weekly_disease.json"

RESOURCE_FILE = DATA_DIR / "facility_resource_allocation.csv"
RESOURCE_SUMMARY_FILE = DATA_DIR / "district_resource_summary.csv"
FACILITY_ALLOCATION_FILE = DATA_DIR / "facility_resource_allocation.csv"


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="EpiScope API",
    description=(
        "Big Data Surveillance, Forecasting and "
        "Response-Planning Platform for Bangladesh"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# DATA LOADING
# ==================================================

@lru_cache(maxsize=1)
def load_json_file(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Data file not found: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array.")

        return data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {path.name}: {error}"
        )


@lru_cache(maxsize=1)
def load_csv_file(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CSV file not found: {path}"
        )

    try:
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            return list(csv.DictReader(file))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {path.name}: {error}"
        )


# ==================================================
# FILTERING AND PAGINATION
# ==================================================

def filter_records(records, district=None, disease=None):
    result = records

    if district:
        result = [
            row for row in result
            if str(row.get("district", "")).lower()
            == district.lower()
        ]

    if disease:
        result = [
            row for row in result
            if str(row.get("disease", "")).lower()
            == disease.lower()
        ]

    return result


def paginate(records, limit, offset):
    total = len(records)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": records[offset:offset + limit]
    }


# ==================================================
# ROOT AND HEALTH
# ==================================================

@app.get("/")
def root():
    return {
        "project": "EpiScope",
        "message": "EpiScope API is running.",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/api/health")
def health():
    files = {
        "district_summary": DISTRICT_FILE,
        "outbreak_alerts": ALERT_FILE,
        "forecasts": FORECAST_FILE,
        "resource_allocation": RESOURCE_FILE,
        "weekly_disease": WEEKLY_FILE,
        "original_alerts": ORIGINAL_ALERT_FILE,
        "resource_summary": RESOURCE_SUMMARY_FILE
    }

    return {
        "status": "healthy",
        "project": "EpiScope",
        "data_files": {
            name: path.exists()
            for name, path in files.items()
        }
    }


# ==================================================
# NATIONAL OVERVIEW
# ==================================================

@app.get("/api/overview")
def overview():
    districts = load_json_file(str(DISTRICT_FILE))
    alerts = load_json_file(str(ALERT_FILE))
    forecasts = load_json_file(str(FORECAST_FILE))

    district_names = {
        row.get("district")
        for row in districts
        if row.get("district")
    }

    total_cases = sum(
        float(row.get("total_cases") or 0)
        for row in districts
    )

    return {
        "project": "EpiScope",
        "total_cases": round(total_cases, 2),
        "total_districts": len(district_names),
        "total_alert_records": len(alerts),
        "total_forecast_records": len(forecasts),
        "forecast_origin": "2025-12-22",
        "data_period": "2021-2025",
        "note": (
            "Historical forecasts and simulated encounter data. "
            "These are not live surveillance figures."
        )
    }


# ==================================================
# DISTRICTS
# ==================================================

@app.get("/api/districts")
def get_districts(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_json_file(str(DISTRICT_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    return paginate(records, limit, offset)


# ==================================================
# IMPROVED OUTBREAK ALERTS
# ==================================================

@app.get("/api/alerts")
def get_alerts(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_json_file(str(ALERT_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    return paginate(records, limit, offset)


# ==================================================
# ORIGINAL OUTBREAK ALERTS
# ==================================================

@app.get("/api/alerts/original")
def get_original_alerts(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_json_file(str(ORIGINAL_ALERT_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    return paginate(records, limit, offset)


# ==================================================
# FORECASTS
# ==================================================

@app.get("/api/forecasts")
def get_forecasts(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    horizon: Optional[int] = Query(None, ge=1, le=4),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_json_file(str(FORECAST_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    if horizon is not None:
        records = [
            row for row in records
            if int(row.get("forecast_horizon", 0)) == horizon
        ]

    return paginate(records, limit, offset)


# ==================================================
# WEEKLY DISEASE TRENDS
# ==================================================

@app.get("/api/trends")
def get_trends(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0)
):
    records = load_json_file(str(WEEKLY_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    return paginate(records, limit, offset)


# ==================================================
# RESOURCE ALLOCATIONS
# ==================================================

@app.get("/api/resources")
def get_resources(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    resource: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_csv_file(str(RESOURCE_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    if resource:
        records = [
            row for row in records
            if row.get("resource", "").lower()
            == resource.lower()
        ]

    return paginate(records, limit, offset)


# ==================================================
# DISTRICT RESOURCE SUMMARY
# ==================================================

@app.get("/api/resource-summary")
def get_resource_summary(
    district: Optional[str] = None,
    resource: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_csv_file(str(RESOURCE_SUMMARY_FILE))

    if district:
        records = [
            row for row in records
            if row.get("district", "").lower()
            == district.lower()
        ]

    if resource:
        records = [
            row for row in records
            if row.get("resource", "").lower()
            == resource.lower()
        ]

    return paginate(records, limit, offset)


# ==================================================
# FACILITY-LEVEL ALLOCATIONS
# ==================================================

@app.get("/api/facility-allocations")
def get_facility_allocations(
    district: Optional[str] = None,
    disease: Optional[str] = None,
    resource: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    records = load_csv_file(str(FACILITY_ALLOCATION_FILE))

    records = filter_records(
        records,
        district=district,
        disease=disease
    )

    if resource:
        records = [
            row for row in records
            if row.get("resource", "").lower()
            == resource.lower()
        ]

    return paginate(records, limit, offset)


# ==================================================
# AVAILABLE DISEASES
# ==================================================

@app.get("/api/diseases")
def get_diseases():
    records = load_json_file(str(DISTRICT_FILE))

    diseases = sorted({
        row.get("disease")
        for row in records
        if row.get("disease")
    })

    return {
        "total": len(diseases),
        "diseases": diseases
    }


# ==================================================
# AVAILABLE DISTRICT NAMES
# ==================================================

@app.get("/api/district-names")
def get_district_names():
    records = load_json_file(str(DISTRICT_FILE))

    districts = sorted({
        row.get("district")
        for row in records
        if row.get("district")
    })

    return {
        "total": len(districts),
        "districts": districts
    }
