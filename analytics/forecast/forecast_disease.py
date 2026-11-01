
from pathlib import Path
from datetime import timedelta

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "weekly_district_disease"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "surge_aware_forecasts"
)

FORECAST_HORIZON = 4
MIN_HISTORY_WEEKS = 52
SEASONAL_LAG = 52


def main():

    spark = (
        SparkSession.builder
        .appName("EpiScope-Surge-Aware-Forecasting")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:

        print("Loading Gold weekly disease data...")

        gold = (
            spark.read.parquet(str(GOLD_DIR))
            .filter(F.col("disease") != "Other")
            .select(
                "district_code",
                "district",
                "disease",
                "week_start",
                "total_cases"
            )
        )

        bounds = gold.agg(
            F.min("week_start").alias("min_week"),
            F.max("week_start").alias("max_week")
        ).first()

        min_week = bounds["min_week"]
        max_week = bounds["max_week"]

        # Exclude the final partial week.
        latest_complete_week = max_week - timedelta(days=7)

        print(f"Historical start: {min_week}")
        print(f"Latest Gold week: {max_week}")
        print(f"Latest complete week: {latest_complete_week}")

        print("Building complete weekly time series...")

        combinations = gold.select(
            "district_code",
            "district",
            "disease"
        ).distinct()

        calendar = spark.createDataFrame(
            [(min_week, latest_complete_week)],
            ["min_week", "max_week"]
        ).select(
            F.explode(
                F.sequence(
                    F.col("min_week"),
                    F.col("max_week"),
                    F.expr("INTERVAL 7 DAYS")
                )
            ).alias("week_start")
        )

        complete = (
            combinations
            .crossJoin(calendar)
            .join(
                gold.filter(
                    F.col("week_start") <= F.lit(latest_complete_week)
                ),
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

        print("Calculating forecasting features...")

        # 12-week window for recent mean and standard deviation.
        history_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-12, -1)
        )

        # Separate 52-week window for eligibility.
        history_count_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-52, -1)
        )

        recent4_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-4, -1)
        )

        previous4_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-8, -5)
        )

        previous8_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-8, -1)
        )

        seasonal_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
        )

        features = (
            complete
            .withColumn(
                "history_weeks",
                F.count("total_cases").over(history_count_window)
            )
            .withColumn(
                "recent_mean",
                F.avg("total_cases").over(history_window)
            )
            .withColumn(
                "recent_stddev",
                F.stddev_pop("total_cases").over(history_window)
            )
            .withColumn(
                "recent4_mean",
                F.avg("total_cases").over(recent4_window)
            )
            .withColumn(
                "previous4_mean",
                F.avg("total_cases").over(previous4_window)
            )
            .withColumn(
                "previous8_mean",
                F.avg("total_cases").over(previous8_window)
            )
            .withColumn(
                "previous8_stddev",
                F.stddev_pop("total_cases").over(previous8_window)
            )
            .withColumn(
                "lag52_cases",
                F.lag("total_cases", SEASONAL_LAG).over(
                    seasonal_window
                )
            )
        )

        print("Calculating trend and seasonal components...")

        features = (
            features
            .withColumn(
                "trend_factor",
                F.when(
                    F.col("previous4_mean") > 0,
                    F.least(
                        F.lit(2.0),
                        F.greatest(
                            F.lit(0.5),
                            F.col("recent4_mean")
                            / F.col("previous4_mean")
                        )
                    )
                )
                .when(
                    F.col("recent4_mean") > 0,
                    F.lit(2.0)
                )
                .otherwise(F.lit(1.0))
            )
            .withColumn(
                "seasonal_cases",
                F.coalesce(
                    F.col("lag52_cases"),
                    F.col("recent_mean")
                )
            )
            .withColumn(
                "trend_seasonal",
                0.4 * F.col("recent_mean")
                + 0.2 * (
                    F.col("recent_mean")
                    * F.col("trend_factor")
                )
                + 0.4 * F.col("seasonal_cases")
            )
        )

        print("Calculating surge warning signals...")

        features = (
            features
            .withColumn(
                "warning_threshold",
                F.col("previous8_mean")
                + 2.0 * F.coalesce(
                    F.col("previous8_stddev"),
                    F.lit(0.0)
                )
            )
            .withColumn(
                "warning_signal",
                F.when(
                    (F.col("recent4_mean") > F.col("warning_threshold"))
                    & (
                        F.col("recent4_mean")
                        > F.col("previous8_mean")
                    ),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            .withColumn(
                "warning_excess",
                F.when(
                    F.col("warning_signal") == 1,
                    F.col("recent4_mean")
                    - F.col("warning_threshold")
                ).otherwise(F.lit(0.0))
            )
        )

        print("Calculating Surge-Aware predictions...")

        features = features.withColumn(
            "surge_aware_prediction",
            F.greatest(
                F.lit(0.0),
                F.least(
                    2.0 * F.col("trend_seasonal"),
                    F.col("trend_seasonal")
                    + 0.5 * F.col("warning_excess")
                )
            )
        )

        print("Selecting latest complete historical week...")

        latest_features = (
            features
            .filter(
                F.col("week_start")
                == F.lit(latest_complete_week)
            )
            .filter(
                F.col("history_weeks") >= MIN_HISTORY_WEEKS
            )
        )

        combination_count = latest_features.count()

        print(
            f"Eligible district-disease combinations: "
            f"{combination_count:,}"
        )

        if combination_count == 0:
            raise ValueError(
                "No eligible combinations. Check historical data."
            )

        print("Generating four-week forecasts...")

        forecast_rows = []

        for horizon in range(1, FORECAST_HORIZON + 1):

            # Immediately after the last complete week.
            calendar_offset = 7 * horizon

            future = latest_features.select(
                "district_code",
                "district",
                "disease",

                F.lit(latest_complete_week)
                .cast("date")
                .alias("origin_week"),

                F.date_add(
                    F.lit(latest_complete_week).cast("date"),
                    calendar_offset
                ).alias("forecast_week"),

                F.lit(horizon).alias("forecast_horizon"),

                "history_weeks",
                "recent_mean",
                "recent_stddev",
                "recent4_mean",
                "previous4_mean",
                "seasonal_cases",
                "trend_factor",
                "warning_threshold",
                "warning_signal",
                "trend_seasonal",
                "surge_aware_prediction"
            )

            forecast_rows.append(future)

        forecast = forecast_rows[0]

        for future in forecast_rows[1:]:
            forecast = forecast.unionByName(future)

        forecast = (
            forecast
            .withColumn(
                "predicted_cases",
                F.round(
                    F.col("surge_aware_prediction")
                ).cast("long")
            )
            .withColumn(
                "lower_bound",
                F.greatest(
                    F.lit(0),
                    F.floor(
                        F.col("predicted_cases")
                        - 1.96 * F.coalesce(
                            F.col("recent_stddev"),
                            F.lit(0.0)
                        )
                    )
                ).cast("long")
            )
            .withColumn(
                "upper_bound",
                F.ceil(
                    F.col("predicted_cases")
                    + 1.96 * F.coalesce(
                        F.col("recent_stddev"),
                        F.lit(0.0)
                    )
                ).cast("long")
            )
            .withColumn(
                "model_name",
                F.lit("Surge-Aware Mean")
            )
            .select(
                "district_code",
                "district",
                "disease",
                "origin_week",
                "forecast_week",
                "forecast_horizon",
                "predicted_cases",
                "lower_bound",
                "upper_bound",
                "recent_mean",
                "recent4_mean",
                "previous4_mean",
                "seasonal_cases",
                "trend_factor",
                "warning_threshold",
                "warning_signal",
                "model_name"
            )
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Writing forecast output...")

        (
            forecast
            .write
            .mode("overwrite")
            .partitionBy("disease")
            .parquet(str(OUTPUT_DIR))
        )

        verified = spark.read.parquet(str(OUTPUT_DIR))

        print("\n========== FORECAST SUMMARY ==========")

        total_records = verified.count()

        print(f"Forecast records: {total_records:,}")
        print(f"Forecast origin: {latest_complete_week}")

        print("\nForecast weeks:")

        verified.select(
            "forecast_week"
        ).distinct().orderBy(
            "forecast_week"
        ).show(FORECAST_HORIZON, truncate=False)

        print("\nForecast totals by disease:")

        verified.groupBy("disease").agg(
            F.sum("predicted_cases").alias("total_predicted_cases"),
            F.avg("predicted_cases").alias("average_cases"),
            F.count("*").alias("forecast_records")
        ).orderBy("disease").show(truncate=False)

        print("\nSurge warning summary:")

        verified.filter(
            F.col("forecast_horizon") == 1
        ).groupBy("disease").agg(
            F.sum("warning_signal").alias("warning_combinations"),
            F.count("*").alias("total_combinations")
        ).orderBy("disease").show(truncate=False)

        print("\nTop predicted district-disease combinations:")

        verified.filter(
            F.col("forecast_horizon") == 1
        ).orderBy(
            F.desc("predicted_cases")
        ).select(
            "district",
            "disease",
            "forecast_week",
            "predicted_cases",
            "lower_bound",
            "upper_bound",
            "warning_signal"
        ).show(20, truncate=False)

        # Validation
        unique_weeks = verified.select(
            "forecast_week"
        ).distinct().count()

        unique_horizons = verified.select(
            "forecast_horizon"
        ).distinct().count()

        duplicate_groups = (
            verified
            .groupBy(
                "district_code",
                "disease",
                "forecast_horizon"
            )
            .count()
            .filter(F.col("count") != 1)
            .count()
        )

        invalid_predictions = verified.filter(
            F.col("predicted_cases").isNull()
            | (F.col("predicted_cases") < 0)
            | F.col("lower_bound").isNull()
            | F.col("upper_bound").isNull()
            | (
                F.col("lower_bound")
                > F.col("upper_bound")
            )
        ).count()

        expected_records = (
            combination_count * FORECAST_HORIZON
        )

        print("\n========== VALIDATION ==========")
        print(f"Expected records: {expected_records:,}")
        print(f"Actual records: {total_records:,}")
        print(f"Unique forecast weeks: {unique_weeks}")
        print(f"Unique forecast horizons: {unique_horizons}")
        print(f"Duplicate groups: {duplicate_groups}")
        print(f"Invalid predictions: {invalid_predictions}")

        if total_records != expected_records:
            raise ValueError("Unexpected forecast record count.")

        if unique_weeks != FORECAST_HORIZON:
            raise ValueError("Expected exactly four forecast weeks.")

        if unique_horizons != FORECAST_HORIZON:
            raise ValueError("Expected four forecast horizons.")

        if duplicate_groups != 0:
            raise ValueError("Duplicate forecast records detected.")

        if invalid_predictions != 0:
            raise ValueError("Invalid forecast values detected.")

        print(f"\nForecast saved to: {OUTPUT_DIR}")
        print("\nSURGE-AWARE FORECASTING COMPLETED SUCCESSFULLY!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
