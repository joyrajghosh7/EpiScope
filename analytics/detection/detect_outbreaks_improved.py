
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "weekly_district_disease"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "outbreak_detection_improved"
)

BASELINE_WEEKS = 8
STDDEV_MULTIPLIER = 2.0
MIN_CASES = 5
MIN_SEASONAL_YEARS = 3


def main():

    spark = (
        SparkSession.builder
        .appName("EpiScope-Improved-Outbreak-Detection")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Loading Gold weekly data...")

        gold = spark.read.parquet(str(GOLD_DIR))

        disease_data = (
            gold
            .filter(F.col("disease") != "Other")
            .select(
                "district_code",
                "district",
                "disease",
                "week_start",
                "total_cases"
            )
        )

        combinations = disease_data.select(
            "district_code",
            "district",
            "disease"
        ).distinct()

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

        complete = (
            combinations
            .crossJoin(calendar)
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
            .withColumn(
                "calendar_year",
                F.year("week_start")
            )
            .withColumn(
                "epi_week",
                F.weekofyear("week_start")
            )
        )

        # -------------------------------------------------
        # BASELINE 1: Previous 8 weeks
        # -------------------------------------------------

        print("Calculating short-term baseline...")

        recent_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-BASELINE_WEEKS, -1)
        )

        scored = (
            complete
            .withColumn(
                "baseline_weeks",
                F.count("total_cases").over(recent_window)
            )
            .withColumn(
                "baseline_mean",
                F.avg("total_cases").over(recent_window)
            )
            .withColumn(
                "baseline_stddev",
                F.stddev_pop("total_cases").over(recent_window)
            )
            .withColumn(
                "recent_threshold",
                F.col("baseline_mean")
                + F.lit(STDDEV_MULTIPLIER)
                * F.col("baseline_stddev")
            )
        )

        # -------------------------------------------------
        # BASELINE 2: Same epidemiological week in previous
        # years. Current year is excluded.
        # -------------------------------------------------

        print("Calculating seasonal baseline...")

        seasonal_window = (
            Window
            .partitionBy(
                "district_code",
                "disease",
                "epi_week"
            )
            .orderBy("calendar_year")
            .rowsBetween(-4, -1)
        )

        scored = (
            scored
            .withColumn(
                "seasonal_baseline_years",
                F.count("total_cases").over(seasonal_window)
            )
            .withColumn(
                "seasonal_mean",
                F.avg("total_cases").over(seasonal_window)
            )
            .withColumn(
                "seasonal_stddev",
                F.stddev_pop("total_cases").over(seasonal_window)
            )
            .withColumn(
                "seasonal_threshold",
                F.col("seasonal_mean")
                + F.lit(STDDEV_MULTIPLIER)
                * F.coalesce(
                    F.col("seasonal_stddev"),
                    F.lit(0.0)
                )
            )
        )

        # -------------------------------------------------
        # COMBINE BOTH DETECTION SIGNALS
        # -------------------------------------------------

        recent_alert = (
            (F.col("baseline_weeks") == BASELINE_WEEKS)
            & (F.col("total_cases") >= MIN_CASES)
            & (F.col("total_cases") > F.col("recent_threshold"))
        )

        seasonal_alert = (
            (F.col("seasonal_baseline_years") >= MIN_SEASONAL_YEARS)
            & (F.col("total_cases") >= MIN_CASES)
            & (F.col("total_cases") > F.col("seasonal_threshold"))
        )

        scored = (
            scored
            .withColumn(
                "recent_alert",
                F.coalesce(recent_alert, F.lit(False))
            )
            .withColumn(
                "seasonal_alert",
                F.coalesce(seasonal_alert, F.lit(False))
            )
            .withColumn(
                "is_outbreak_alert",
                F.col("recent_alert") | F.col("seasonal_alert")
            )
            .withColumn(
                "alert_reason",
                F.when(
                    F.col("recent_alert") & F.col("seasonal_alert"),
                    "Recent and seasonal anomaly"
                )
                .when(
                    F.col("recent_alert"),
                    "Recent 8-week anomaly"
                )
                .when(
                    F.col("seasonal_alert"),
                    "Historical seasonal anomaly"
                )
                .otherwise("No alert")
            )
            .withColumn(
                "detection_method",
                F.lit("Dual baseline: 8-week rolling + same epi-week historical")
            )
        )

        # -------------------------------------------------
        # SAVE SCORES AND ALERTS
        # -------------------------------------------------

        scores_path = OUTPUT_DIR / "weekly_scores"
        alerts_path = OUTPUT_DIR / "alerts"

        print("Writing improved weekly scores...")

        (
            scored
            .drop("calendar_year", "epi_week")
            .write
            .mode("overwrite")
            .partitionBy("disease")
            .parquet(str(scores_path))
        )

        print("Writing improved alerts...")

        alerts = scored.filter(
            F.col("is_outbreak_alert")
        ).drop("calendar_year", "epi_week")

        (
            alerts
            .write
            .mode("overwrite")
            .parquet(str(alerts_path))
        )

        # -------------------------------------------------
        # VERIFY
        # -------------------------------------------------

        verified_scores = spark.read.parquet(str(scores_path))
        verified_alerts = spark.read.parquet(str(alerts_path))

        print("\n========== IMPROVED DETECTION SUMMARY ==========")

        print(f"Weekly scores: {verified_scores.count():,}")
        print(f"Alert weeks: {verified_alerts.count():,}")

        print("\nAlerts by disease:")

        verified_alerts.groupBy("disease").agg(
            F.count("*").alias("alert_weeks"),
            F.countDistinct("district_code").alias("districts_affected")
        ).orderBy("disease").show(truncate=False)

        print("\nAlerts by reason:")

        verified_alerts.groupBy("alert_reason").count().show(
            truncate=False
        )

        print("\nTop detected alerts:")

        verified_alerts.orderBy(
            F.desc("total_cases")
        ).select(
            "district",
            "disease",
            "week_start",
            "total_cases",
            "baseline_mean",
            "seasonal_mean",
            "seasonal_threshold",
            "alert_reason"
        ).show(20, truncate=False)

        print("\nIMPROVED OUTBREAK DETECTION COMPLETED!")

        print(f"Output: {OUTPUT_DIR}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
