
import json
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "dashboard" / "static" / "data"

OUTPUT.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("EpiScopeDashboardExporter")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def export_json(df, filename):
    """Write a Spark DataFrame as one JSON array."""
    records = [
        row.asDict(recursive=True)
        for row in df.collect()
    ]

    output_path = OUTPUT / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2, default=str)

    print(f"Exported {filename}: {len(records):,} records")


def export_district_summary(weekly):
    summary = (
        weekly
        .groupBy("district_code", "district", "disease")
        .agg(
            F.sum("total_cases").alias("total_cases"),
            F.sum("admissions").alias("admissions"),
            F.sum("lab_confirmed_cases").alias("lab_confirmed_cases"),
            F.avg("cases_per_100k").alias("avg_cases_per_100k"),
            F.avg("avg_temperature_c").alias("avg_temperature_c"),
            F.avg("avg_humidity_percent").alias("avg_humidity_percent"),
            F.sum("weekly_rainfall_mm").alias("total_rainfall_mm")
        )
    )

    export_json(summary.orderBy("district", "disease"), "district_summary.json")


def main():
    print("\nLoading dashboard sources...")

    weekly = spark.read.parquet(
        str(ROOT / "data" / "gold" / "weekly_district_disease")
    )

    original = spark.read.parquet(
        str(ROOT / "data" / "gold" / "outbreak_detection" / "alerts")
    )

    improved = spark.read.parquet(
        str(ROOT / "data" / "gold" / "outbreak_detection_improved" / "alerts")
    )

    forecasts = spark.read.parquet(
        str(ROOT / "data" / "gold" / "surge_aware_forecasts")
    )

    print("\nExporting dashboard datasets...")

    # Weekly disease trends
    export_json(
        weekly.orderBy("week_start", "district", "disease"),
        "weekly_disease.json"
    )

    # Original detector alerts
    export_json(
        original.orderBy("week_start", "district", "disease"),
        "outbreak_alerts_original.json"
    )

    # Improved detector alerts
    export_json(
        improved.orderBy("week_start", "district", "disease"),
        "outbreak_alerts_improved.json"
    )

    # Four-week forecasts
    export_json(
        forecasts.orderBy("forecast_week", "district", "disease"),
        "forecasts.json"
    )

    # District-level aggregated statistics
    export_district_summary(weekly)

    # Latest resource allocation outputs
    resource_dir = ROOT / "data" / "gold" / "resource_optimisation"

    for filename in [
        "facility_resource_allocation.csv",
        "district_resource_summary.csv",
        "optimisation_metadata.json"
    ]:
        source = resource_dir / filename

        if source.exists():
            shutil.copy2(source, OUTPUT / filename)
            print(f"Copied {filename}")
        else:
            print(f"WARNING: Missing {filename}")

    print("\nDASHBOARD DATA EXPORT COMPLETED")
    print("Output directory:", OUTPUT)

    for file in sorted(OUTPUT.iterdir()):
        if file.is_file():
            print(f"{file.name}: {file.stat().st_size / 1024:.1f} KB")

    spark.stop()


if __name__ == "__main__":
    main()
