
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "weekly_district_disease"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold" / "outbreak_detection"

BASELINE_WEEKS = 8
STDDEV_MULTIPLIER = 2.0
MIN_CASES = 5


def main():
    spark = (
        SparkSession.builder
        .appName("EpiScope-Outbreak-Detection")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Loading Gold weekly data...")

        gold = spark.read.parquet(str(GOLD_DIR))

        # Exclude the general background category from outbreak alerts.
        disease_data = gold.filter(
            F.col("disease") != "Other"
        ).select(
            "district_code",
            "district",
            "disease",
            "week_start",
            "total_cases"
        )

        # Create district-disease combinations.
        combinations = disease_data.select(
            "district_code", "district", "disease"
        ).distinct()

        # Generate a complete calendar of epidemiological weeks.
        bounds = disease_data.agg(
            F.min("week_start").alias("min_week"),
            F.max("week_start").alias("max_week")
        )

        calendar = bounds.select(
            F.explode(
                F.sequence(
                    F.col("min_week"),
                    F.col("max_week"),
                    F.expr("INTERVAL 7 DAYS")
                )
            ).alias("week_start")
        )

        print("Building complete weekly calendar...")

        complete = combinations.crossJoin(calendar)

        # Fill weeks with no recorded cases with zero.
        complete = (
            complete
            .join(
                disease_data,
                on=[
                    "district_code",
                    "district",
                    "disease",
                    "week_start"
                ],
                how="left"
            )
            .withColumn(
                "total_cases",
                F.coalesce(F.col("total_cases"), F.lit(0))
            )
        )

        # Historical baseline: previous 8 calendar weeks,
        # excluding the current week.
        history_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-BASELINE_WEEKS, -1)
        )

        print("Calculating historical baselines...")

        scored = (
            complete
            .withColumn(
                "baseline_weeks",
                F.count("total_cases").over(history_window)
            )
            .withColumn(
                "baseline_mean",
                F.avg("total_cases").over(history_window)
            )
            .withColumn(
                "baseline_stddev",
                F.stddev_pop("total_cases").over(history_window)
            )
            .withColumn(
                "alert_threshold",
                F.col("baseline_mean")
                + F.lit(STDDEV_MULTIPLIER) * F.col("baseline_stddev")
            )
            .withColumn(
                "z_score",
                F.when(
                    F.col("baseline_stddev") > 0,
                    (
                        F.col("total_cases") - F.col("baseline_mean")
                    ) / F.col("baseline_stddev")
                )
            )
            .withColumn(
                "is_outbreak_alert",
                (
                    (F.col("baseline_weeks") == BASELINE_WEEKS)
                    & (F.col("total_cases") >= MIN_CASES)
                    & (F.col("total_cases") > F.col("alert_threshold"))
                )
            )
            .withColumn(
                "detection_method",
                F.lit("8-week rolling mean + 2 standard deviations")
            )
        )

        # Store all weekly scores for analysis and dashboard use.
        scores_path = OUTPUT_DIR / "weekly_scores"

        print("Writing weekly detection scores...")

        (
            scored
            .write
            .mode("overwrite")
            .partitionBy("disease")
            .parquet(str(scores_path))
        )

        # Store only flagged district-disease weeks as alerts.
        alerts = scored.filter(
            F.col("is_outbreak_alert") == True
        )

        alerts_path = OUTPUT_DIR / "alerts"

        print("Writing outbreak alerts...")

        (
            alerts
            .write
            .mode("overwrite")
            .parquet(str(alerts_path))
        )

        # Verify the outputs.
        verified_scores = spark.read.parquet(str(scores_path))
        verified_alerts = spark.read.parquet(str(alerts_path))

        score_count = verified_scores.count()
        alert_count = verified_alerts.count()

        print("\n========== DETECTION SUMMARY ==========")
        print(f"Weekly district-disease scores: {score_count:,}")
        print(f"Outbreak alert records: {alert_count:,}")
        print(f"Baseline weeks: {BASELINE_WEEKS}")
        print(f"Standard deviation multiplier: {STDDEV_MULTIPLIER}")
        print(f"Minimum cases: {MIN_CASES}")
        print(f"Scores output: {scores_path}")
        print(f"Alerts output: {alerts_path}")

        print("\nAlerts by disease:")
        verified_alerts.groupBy("disease").agg(
            F.count("*").alias("alert_weeks"),
            F.countDistinct("district_code").alias("districts_affected")
        ).orderBy("disease").show(truncate=False)

        print("\nTop detected alerts:")
        verified_alerts.orderBy(
            F.desc("z_score"),
            F.desc("total_cases")
        ).select(
            "district",
            "disease",
            "week_start",
            "total_cases",
            "baseline_mean",
            "baseline_stddev",
            "alert_threshold",
            "z_score"
        ).show(20, truncate=False)

        print("\nOUTBREAK DETECTION COMPLETED SUCCESSFULLY!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
