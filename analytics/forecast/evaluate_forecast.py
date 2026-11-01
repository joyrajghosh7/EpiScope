
from pathlib import Path
from datetime import timedelta

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "weekly_district_disease"

PREDICTIONS_DIR = (
    PROJECT_ROOT / "data" / "gold" / "forecast_backtest" / "predictions"
)

METRICS_DIR = (
    PROJECT_ROOT / "data" / "gold" / "forecast_backtest" / "metrics"
)

FORECAST_HORIZON = 4
MIN_HISTORY_WEEKS = 12
SEASONAL_LAG = 52


def main():

    spark = (
        SparkSession.builder
        .appName("EpiScope-Forecast-Backtesting")
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

        print(f"Historical start: {min_week}")
        print(f"Historical end: {max_week}")

        # Build a complete weekly calendar.
        print("Building complete historical time series...")

        combinations = gold.select(
            "district_code",
            "district",
            "disease"
        ).distinct()

        calendar = spark.createDataFrame(
            [(min_week, max_week)],
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
                gold,
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

        # Calculate features using only observations before each origin.
        print("Calculating rolling historical features...")

        history_window = (
            Window
            .partitionBy("district_code", "disease")
            .orderBy("week_start")
            .rowsBetween(-MIN_HISTORY_WEEKS, -1)
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
                F.count("total_cases").over(history_window)
            )
            .withColumn(
                "recent_mean",
                F.avg("total_cases").over(history_window)
            )
            .withColumn(
                "seasonal_cases",
                F.lag("total_cases", SEASONAL_LAG).over(
                    seasonal_window
                )
            )
        )

        # A backtest origin must have:
        # 1. At least 52 weeks of seasonal history.
        # 2. At least 12 weeks of recent history.
        # 3. Four future weeks of actual data available.
        first_origin = min_week + timedelta(days=7 * SEASONAL_LAG)
        last_origin = max_week - timedelta(days=7 * FORECAST_HORIZON)

        print(f"First eligible origin: {first_origin}")
        print(f"Last eligible origin: {last_origin}")

        origins = (
            features
            .filter(
                (F.col("week_start") >= F.lit(first_origin))
                & (F.col("week_start") <= F.lit(last_origin))
                & (F.col("history_weeks") >= MIN_HISTORY_WEEKS)
                & F.col("seasonal_cases").isNotNull()
            )
            .select(
                "district_code",
                "district",
                "disease",
                F.col("week_start").alias("forecast_origin"),
                "recent_mean",
                "seasonal_cases"
            )
        )

        print("Generating historical predictions...")

        forecast_rows = []

        for horizon in range(1, FORECAST_HORIZON + 1):

            future = origins.select(
                "district_code",
                "district",
                "disease",
                "forecast_origin",
                F.date_add(
                    F.col("forecast_origin"),
                    7 * horizon
                ).alias("forecast_week"),
                F.lit(horizon).alias("forecast_horizon"),
                "recent_mean",
                "seasonal_cases"
            )

            forecast_rows.append(future)

        base_predictions = forecast_rows[0]

        for future in forecast_rows[1:]:
            base_predictions = base_predictions.unionByName(future)

        # Model 1: Recent 12-week average.
        recent_model = (
            base_predictions
            .select(
                "*",
                F.col("recent_mean").alias("raw_prediction")
            )
            .withColumn(
                "model_name",
                F.lit("Recent Mean")
            )
        )

        # Model 2: 60% recent mean + 40% seasonal cases.
        seasonal_model = (
            base_predictions
            .select(
                "*",
                (
                    0.6 * F.col("recent_mean")
                    + 0.4 * F.col("seasonal_cases")
                ).alias("raw_prediction")
            )
            .withColumn(
                "model_name",
                F.lit("Seasonal-Recent Mean")
            )
        )

        predictions = (
            recent_model
            .unionByName(seasonal_model)
            .withColumn(
                "predicted_cases",
                F.round(
                    F.greatest(
                        F.lit(0.0),
                        F.col("raw_prediction")
                    )
                ).cast("long")
            )
            .drop("raw_prediction")
        )

        # Attach actual future case counts.
        actuals = gold.select(
            "district_code",
            "disease",
            F.col("week_start").alias("forecast_week"),
            F.col("total_cases").alias("actual_cases")
        )

        evaluated = (
            predictions
            .join(
                actuals,
                on=[
                    "district_code",
                    "disease",
                    "forecast_week"
                ],
                how="inner"
            )
            .withColumn(
                "absolute_error",
                F.abs(
                    F.col("actual_cases")
                    - F.col("predicted_cases")
                )
            )
            .withColumn(
                "squared_error",
                F.pow(
                    F.col("actual_cases")
                    - F.col("predicted_cases"),
                    2
                )
            )
        )

        print("Calculating evaluation metrics...")

        # Metrics for each disease and forecast horizon.
        metrics = (
            evaluated
            .groupBy(
                "model_name",
                "disease",
                "forecast_horizon"
            )
            .agg(
                F.count("*").alias("evaluated_predictions"),
                F.avg("absolute_error").alias("MAE"),
                F.sqrt(
                    F.avg("squared_error")
                ).alias("RMSE"),
                (
                    F.sum("absolute_error")
                    / F.when(
                        F.sum("actual_cases") > 0,
                        F.sum("actual_cases")
                    ).otherwise(F.lit(None))
                    * 100
                ).alias("WAPE_percent")
            )
            .orderBy(
                "model_name",
                "disease",
                "forecast_horizon"
            )
        )

        # Overall metrics for each model and horizon.
        overall_metrics = (
            evaluated
            .groupBy(
                "model_name",
                "forecast_horizon"
            )
            .agg(
                F.count("*").alias("evaluated_predictions"),
                F.avg("absolute_error").alias("MAE"),
                F.sqrt(
                    F.avg("squared_error")
                ).alias("RMSE"),
                (
                    F.sum("absolute_error")
                    / F.when(
                        F.sum("actual_cases") > 0,
                        F.sum("actual_cases")
                    ).otherwise(F.lit(None))
                    * 100
                ).alias("WAPE_percent")
            )
            .orderBy(
                "forecast_horizon",
                "model_name"
            )
        )

        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

        print("Saving backtest predictions...")

        (
            evaluated.write
            .mode("overwrite")
            .parquet(str(PREDICTIONS_DIR))
        )

        print("Saving evaluation metrics...")

        (
            metrics.write
            .mode("overwrite")
            .parquet(str(METRICS_DIR / "by_disease"))
        )

        (
            overall_metrics.write
            .mode("overwrite")
            .parquet(str(METRICS_DIR / "overall"))
        )

        print("\n========== BACKTEST SUMMARY ==========")

        print(
            f"Evaluated prediction records: {evaluated.count():,}"
        )

        print(
            f"Forecast origins: "
            f"{origins.select('forecast_origin').distinct().count():,}"
        )

        print("\nOVERALL MODEL PERFORMANCE:")

        overall_metrics.show(20, truncate=False)

        print("\nPERFORMANCE BY DISEASE:")

        metrics.show(40, truncate=False)

        print("\n========== OUTPUT LOCATIONS ==========")
        print(f"Predictions: {PREDICTIONS_DIR}")
        print(f"Metrics: {METRICS_DIR}")

        print("\nFORECAST EVALUATION COMPLETED SUCCESSFULLY!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
