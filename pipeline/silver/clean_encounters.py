
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Start with the sample. We will switch to full Bronze after testing.
INPUT_DIR = PROJECT_ROOT / "data" / "bronze" / "encounters_parquet"
FACILITY_FILE = PROJECT_ROOT / "data" / "bronze" / "facilities" / "facilities.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "silver" / "encounters"
QUARANTINE_DIR = (
    PROJECT_ROOT / "data" / "silver" / "quarantine" / "encounters"
)

HASH_SALT = os.environ.get("EPISCOPE_HASH_SALT")

ALLOWED_CODES = [
    "A90", "A91", "A00", "A09", "J10", "J11",
    "B50", "B51", "B52", "B53", "B54",
    "Z00", "R50", "B99"
]


def main():
    if not HASH_SALT or len(HASH_SALT) < 32:
        raise ValueError(
            "Set EPISCOPE_HASH_SALT to a private secret of at least 32 characters."
        )

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("EpiScope-Silver-Cleaning")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("=" * 60)
        print("EPISCOPE SILVER DATA CLEANING")
        print("=" * 60)

        print("\nReading Bronze encounters...")
        df = spark.read.parquet(str(INPUT_DIR))
        original_count = df.count()
        print("Bronze records:", f"{original_count:,}")

        print("\nReading facility registry...")
        facilities = (
            spark.read
            .option("header", True)
            .schema("facility_id STRING")
            .csv(str(FACILITY_FILE))
            .select("facility_id")
            .dropDuplicates()
            .withColumn("facility_exists", F.lit(True))
        )

        # Parse and validate source fields.
        df = (
            df
            .withColumn("age_years", F.col("age_years").cast("int"))
            .withColumn("onset_date", F.to_date("onset_date"))
            .withColumn("report_date", F.to_date("report_date"))
            .withColumn(
                "reporting_delay_days",
                F.datediff("report_date", "onset_date")
            )
        )

        df = df.join(facilities, on="facility_id", how="left")

        # Assign one clear reason to each invalid record.
        df = df.withColumn(
            "quarantine_reason",
            F.when(
                F.col("age_years").isNull()
                | (F.col("age_years") < 0)
                | (F.col("age_years") > 120),
                "INVALID_AGE"
            )
            .when(
                F.col("sex").isNull()
                | (~F.col("sex").isin("Male", "Female")),
                "INVALID_SEX"
            )
            .when(
                F.col("facility_exists").isNull(),
                "UNKNOWN_FACILITY"
            )
            .when(
                F.col("onset_date").isNull()
                | F.col("report_date").isNull(),
                "INVALID_DATE"
            )
            .when(
                F.col("reporting_delay_days") < 0,
                "NEGATIVE_REPORTING_DELAY"
            )
            .when(
                F.col("reporting_delay_days") > 21,
                "REPORTING_DELAY_OVER_21_DAYS"
            )
            .when(
                F.col("icd10_code").isNull()
                | (~F.col("icd10_code").isin(*ALLOWED_CODES)),
                "INVALID_DISEASE_CODE"
            )
            .when(
                F.col("patient_id").isNull()
                | (F.length(F.trim(F.col("patient_id"))) == 0),
                "MISSING_PATIENT_ID"
            )
            .otherwise(None)
        )

        quarantine = df.filter(F.col("quarantine_reason").isNotNull())
        clean = df.filter(F.col("quarantine_reason").isNull())

        print("\nCounting quarantined records...")
        quarantine_count = quarantine.count()
        print("Quarantined records:", f"{quarantine_count:,}")

        # Never store raw patient IDs in the quarantine output.
        quarantine_output = quarantine.drop(
            "patient_id", "facility_exists"
        )

        print("\nWriting quarantine records...")
        (
            quarantine_output.write
            .mode("overwrite")
            .parquet(str(QUARANTINE_DIR))
        )

        # Keep one valid row per encounter ID.
        window = Window.partitionBy("encounter_id").orderBy(
            F.col("report_date").asc(),
            F.col("facility_id").asc()
        )

        clean = (
            clean
            .withColumn(
                "duplicate_rank",
                F.row_number().over(window)
            )
            .filter(F.col("duplicate_rank") == 1)
            .drop("duplicate_rank")
        )

        # Hash patient ID before removing the original identifier.
        clean = clean.withColumn(
            "patient_hash",
            F.sha2(
                F.concat(
                    F.lit(HASH_SALT),
                    F.col("patient_id")
                ),
                256
            )
        )

        # Keep parsed age and date fields in Silver.
        clean = clean.drop(
            "patient_id",
            "facility_exists",
            "quarantine_reason"
        )

        clean_count = clean.count()

        print("\nWriting clean Silver records...")
        (
            clean.write
            .mode("overwrite")
            .partitionBy("district")
            .parquet(str(OUTPUT_DIR))
        )

        print("\nVerifying Silver output...")
        result = spark.read.parquet(str(OUTPUT_DIR))
        result_count = result.count()

        print("\n" + "=" * 60)
        print("SILVER CLEANING SUMMARY")
        print("=" * 60)
        print("Original Bronze records:", f"{original_count:,}")
        print("Quarantined records:", f"{quarantine_count:,}")
        print("Clean unique records:", f"{clean_count:,}")
        print("Verified Silver records:", f"{result_count:,}")
        print("Output:", OUTPUT_DIR)
        print("Quarantine:", QUARANTINE_DIR)

        if result_count != clean_count:
            raise RuntimeError("Silver record count mismatch!")

        if "patient_id" in result.columns:
            raise RuntimeError("Raw patient IDs found in Silver!")

        if "patient_hash" not in result.columns:
            raise RuntimeError("Patient hash missing from Silver!")

        print("\nSILVER CLEANING SUCCESSFUL!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
