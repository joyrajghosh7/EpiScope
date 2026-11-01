
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "encounters"
DISTRICTS_FILE = PROJECT_ROOT / "data" / "reference" / "districts.csv"
POPULATION_FILE = PROJECT_ROOT / "data" / "reference" / "district_population_totals.csv"
WEATHER_FILE = PROJECT_ROOT / "data" / "bronze" / "weather" / "weather_daily.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "gold" / "weekly_district_disease"


def main():
    spark = (
        SparkSession.builder
        .appName("EpiScope-Weekly-Gold-Aggregation")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("\nLoading Silver encounters...")
        encounters = spark.read.parquet(str(SILVER_DIR))

        print("Loading district reference...")
        districts = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(str(DISTRICTS_FILE))
            .select(
                "district_code",
                F.lower(F.trim(F.col("district"))).alias("district_key")
            )
            .dropDuplicates(["district_key"])
        )

        print("Loading WorldPop population...")
        population = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(str(POPULATION_FILE))
            .select(
                "district_code",
                F.col("total_population").cast("double").alias("total_population"),
                F.col("year").cast("int").alias("population_year"),
                "data_source"
            )
        )

        print("Loading daily weather...")
        weather_raw = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(str(WEATHER_FILE))
        )

        # Convert district names to district codes.
        encounters = encounters.withColumn(
            "district_key",
            F.lower(F.trim(F.col("district")))
        )

        encounters = encounters.join(
            districts,
            on="district_key",
            how="left"
        )

        unmapped_count = encounters.filter(
            F.col("district_code").isNull()
        ).count()

        if unmapped_count > 0:
            print(f"ERROR: {unmapped_count} encounters have unmapped districts.")
            encounters.filter(
                F.col("district_code").isNull()
            ).select("district").distinct().show(20, False)
            raise RuntimeError("District mapping failed.")

        # Map ICD-10 codes to disease groups.
        code = F.col("icd10_code")

        encounters = encounters.withColumn(
            "disease",
            F.when(code.isin("A90", "A91"), "Dengue")
            .when(code.isin("A00", "A09"), "Diarrhoea")
            .when(code.isin("J10", "J11"), "Influenza")
            .when(code.isin("B50", "B51", "B52", "B53", "B54"), "Malaria")
            .otherwise("Other")
        )

        # Epidemiological week starts on Monday.
        encounters = encounters.withColumn(
            "week_start",
            F.to_date(F.date_trunc("week", F.col("onset_date")))
        )

        encounters = encounters.withColumn(
            "epi_year",
            F.year(F.date_add(F.col("week_start"), 3))
        )

        print("Aggregating weekly disease statistics...")
        weekly_cases = (
            encounters
            .groupBy(
                "district_code",
                "district",
                "disease",
                "week_start",
                "epi_year"
            )
            .agg(
                F.count("*").alias("total_cases"),
                F.countDistinct("patient_hash").alias("unique_patients"),
                F.sum(
                    F.when(F.col("admitted") == True, 1).otherwise(0)
                ).alias("admissions"),
                F.sum(
                    F.when(F.col("lab_confirmed") == True, 1).otherwise(0)
                ).alias("lab_confirmed_cases")
            )
        )

        # Convert daily weather into weekly district-level measurements.
        print("Aggregating weekly weather...")
        weather = (
            weather_raw
            .withColumn("date", F.to_date("date"))
            .withColumn(
                "week_start",
                F.to_date(F.date_trunc("week", F.col("date")))
            )
            .groupBy("district_code", "week_start")
            .agg(
                F.sum("precipitation_mm").alias("weekly_rainfall_mm"),
                F.avg("temperature_mean_c").alias("avg_temperature_c"),
                F.avg("humidity_mean_percent").alias("avg_humidity_percent")
            )
        )

        # Join disease statistics with weather and population.
        print("Joining population and weather...")
        gold = (
            weekly_cases
            .join(
                weather,
                on=["district_code", "week_start"],
                how="left"
            )
            .join(
                population,
                on="district_code",
                how="left"
            )
        )

        missing_population = gold.filter(
            F.col("total_population").isNull()
            | (F.col("total_population") <= 0)
        ).count()

        if missing_population > 0:
            raise RuntimeError(
                f"{missing_population} Gold rows have missing population."
            )

        gold = gold.withColumn(
            "cases_per_100k",
            F.round(
                F.col("total_cases") * 100000.0
                / F.col("total_population"),
                4
            )
        )

        # Keep the population reference year explicit.
        gold = gold.withColumn(
            "population_reference",
            F.lit("WorldPop 2022 estimate")
        )

        output_columns = [
            "district_code",
            "district",
            "disease",
            "week_start",
            "epi_year",
            "total_cases",
            "unique_patients",
            "admissions",
            "lab_confirmed_cases",
            "weekly_rainfall_mm",
            "avg_temperature_c",
            "avg_humidity_percent",
            "total_population",
            "population_year",
            "population_reference",
            "data_source",
            "cases_per_100k"
        ]

        gold = gold.select(*output_columns)

        print("Checking weather coverage...")
        missing_weather = gold.filter(
            F.col("weekly_rainfall_mm").isNull()
            | F.col("avg_temperature_c").isNull()
            | F.col("avg_humidity_percent").isNull()
        ).count()

        print(f"Gold rows missing weather: {missing_weather}")

        print("Writing Gold Parquet table...")
        (
            gold
            .repartition(16, "epi_year", "disease")
            .write
            .mode("overwrite")
            .partitionBy("epi_year", "disease")
            .parquet(str(OUTPUT_DIR))
        )

        print("Verifying Gold output...")
        verified = spark.read.parquet(str(OUTPUT_DIR))
        gold_count = verified.count()

        print("\n========== GOLD SUMMARY ==========")
        print(f"Gold records: {gold_count:,}")
        print(f"Unmapped districts: {unmapped_count:,}")
        print(f"Rows missing population: {missing_population:,}")
        print(f"Rows missing weather: {missing_weather:,}")
        print(f"Output: {OUTPUT_DIR}")

        print("\nDisease summary:")
        verified.groupBy("disease").agg(
            F.sum("total_cases").alias("cases"),
            F.count("*").alias("district_weeks")
        ).orderBy("disease").show(truncate=False)

        print("Sample Gold records:")
        verified.orderBy(
            "week_start", "district", "disease"
        ).show(10, truncate=False)

        if gold_count == 0:
            raise RuntimeError("Gold output is empty.")

        print("\nGOLD AGGREGATION SUCCESSFUL!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
