
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "encounters"
    / "encounters_test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "encounters_parquet"
)


def main():
    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("EpiScope-Bronze-Ingestion")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 55)
    print("EPISCOPE BRONZE INGESTION")
    print("=" * 55)

    print("\nReading encounter CSV...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")
        .csv(str(INPUT_FILE))
    )

    print("\nInput columns:")
    print(df.columns)

    print("\nCounting input records...")
    total_records = df.count()

    print("Total records:", f"{total_records:,}")

    print("\nWriting Parquet files...")

    (
        df.write
        .mode("overwrite")
        .parquet(str(OUTPUT_DIR))
    )

    print("\nVerifying Parquet output...")

    parquet_df = spark.read.parquet(str(OUTPUT_DIR))
    parquet_count = parquet_df.count()

    print("Parquet records:", f"{parquet_count:,}")

    if total_records != parquet_count:
        raise RuntimeError("Bronze record count mismatch!")

    print("\nBRONZE INGESTION SUCCESSFUL!")
    print("Output:", OUTPUT_DIR)

    spark.stop()


if __name__ == "__main__":
    main()
