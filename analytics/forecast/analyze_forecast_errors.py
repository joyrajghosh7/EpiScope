
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT / "data" / "gold" / "forecast_backtest" / "predictions"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "gold" / "forecast_error_analysis"
)

MODEL_NAME = "Seasonal-Recent Mean"

# Exploratory definition: actual cases are at least 50% higher
# than predicted cases, with actual cases greater than prediction.
SURGE_MULTIPLIER = 1.5


def add_error_columns(df):

    return (
        df
        .withColumn(
            "signed_error",
            F.col("actual_cases") - F.col("predicted_cases")
        )
        .withColumn(
            "absolute_error",
            F.abs(F.col("signed_error"))
        )
        .withColumn(
            "squared_error",
            F.pow(F.col("signed_error"), 2)
        )
        .withColumn(
            "underpredicted",
            F.col("actual_cases") > F.col("predicted_cases")
        )
        .withColumn(
            "overpredicted",
            F.col("actual_cases") < F.col("predicted_cases")
        )
        .withColumn(
            "surge_ratio",
            F.when(
                F.col("predicted_cases") > 0,
                F.col("actual_cases") / F.col("predicted_cases")
            )
        )
    )


def calculate_metrics(df, group_columns):

    return (
        df
        .groupBy(*group_columns)
        .agg(
            F.count("*").alias("observations"),
            F.sum("actual_cases").alias("total_actual_cases"),
            F.sum("predicted_cases").alias("total_predicted_cases"),
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
            ).alias("WAPE_percent"),
            F.sum(
                F.when(
                    F.col("underpredicted"),
                    1
                ).otherwise(0)
            ).alias("underpredicted_count"),
            F.avg(
                F.col("underpredicted").cast("double")
            ).alias("underprediction_rate")
        )
    )


def main():

    spark = (
        SparkSession.builder
        .appName("EpiScope-Forecast-Error-Analysis")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:

        print("Loading backtesting predictions...")

        predictions = (
            spark.read.parquet(str(INPUT_DIR))
            .filter(F.col("model_name") == MODEL_NAME)
        )

        total_records = predictions.count()

        print(f"Loaded predictions: {total_records:,}")
        print(f"Selected model: {MODEL_NAME}")

        if total_records == 0:
            raise ValueError("No predictions found for the selected model.")

        data = add_error_columns(predictions)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Overall performance by disease.
        print("\nAnalyzing disease-level errors...")

        disease_errors = (
            calculate_metrics(data, ["disease"])
            .orderBy(F.desc("WAPE_percent"))
        )

        # 2. Districts with the highest forecasting errors.
        print("Analyzing district-level errors...")

        district_errors = (
            calculate_metrics(
                data,
                ["district_code", "district"]
            )
            .orderBy(F.desc("WAPE_percent"))
        )

        # 3. Error growth across forecast horizons.
        print("Analyzing forecast horizons...")

        horizon_errors = (
            calculate_metrics(data, ["forecast_horizon"])
            .orderBy("forecast_horizon")
        )

        # 4. Identify the largest individual prediction errors.
        print("Finding largest individual prediction errors...")

        largest_errors = (
            data
            .select(
                "district_code",
                "district",
                "disease",
                "forecast_origin",
                "forecast_week",
                "forecast_horizon",
                "actual_cases",
                "predicted_cases",
                "signed_error",
                "absolute_error",
                "surge_ratio"
            )
            .orderBy(F.desc("absolute_error"))
            .limit(100)
        )

        # 5. Analyze weeks with large underpredictions.
        print("Analyzing potential disease surges...")

        surge_rows = (
            data
            .filter(
                (F.col("actual_cases") > F.col("predicted_cases"))
                & (
                    F.col("actual_cases")
                    >= F.col("predicted_cases") * SURGE_MULTIPLIER
                )
            )
            .select(
                "district_code",
                "district",
                "disease",
                "forecast_origin",
                "forecast_week",
                "forecast_horizon",
                "actual_cases",
                "predicted_cases",
                "signed_error",
                "surge_ratio"
            )
            .orderBy(F.desc("signed_error"))
        )

        # 6. Aggregate district-level surges by disease and week.
        weekly_surge_analysis = (
            data
            .groupBy(
                "disease",
                "forecast_week",
                "forecast_horizon"
            )
            .agg(
                F.sum("actual_cases").alias("actual_cases"),
                F.sum("predicted_cases").alias("predicted_cases"),
                F.sum("absolute_error").alias("total_absolute_error")
            )
            .withColumn(
                "underprediction",
                F.col("actual_cases") - F.col("predicted_cases")
            )
            .withColumn(
                "surge_ratio",
                F.when(
                    F.col("predicted_cases") > 0,
                    F.col("actual_cases") / F.col("predicted_cases")
                )
            )
            .withColumn(
                "surge_detected",
                (
                    (F.col("actual_cases") > F.col("predicted_cases"))
                    & (
                        F.col("actual_cases")
                        >= F.col("predicted_cases") * SURGE_MULTIPLIER
                    )
                )
            )
            .orderBy(F.desc("underprediction"))
        )

        # 7. Overall underprediction and overprediction statistics.
        direction_summary = (
            data
            .groupBy("forecast_horizon")
            .agg(
                F.count("*").alias("observations"),
                F.sum(
                    F.when(F.col("underpredicted"), 1).otherwise(0)
                ).alias("underpredicted"),
                F.sum(
                    F.when(F.col("overpredicted"), 1).otherwise(0)
                ).alias("overpredicted"),
                F.sum(
                    F.when(
                        F.col("signed_error") == 0,
                        1
                    ).otherwise(0)
                ).alias("exact_predictions")
            )
            .withColumn(
                "underprediction_percent",
                F.col("underpredicted") / F.col("observations") * 100
            )
            .withColumn(
                "overprediction_percent",
                F.col("overpredicted") / F.col("observations") * 100
            )
            .orderBy("forecast_horizon")
        )

        # Save all analysis outputs.
        outputs = {
            "disease_errors": disease_errors,
            "district_errors": district_errors,
            "horizon_errors": horizon_errors,
            "largest_errors": largest_errors,
            "surge_rows": surge_rows,
            "weekly_surge_analysis": weekly_surge_analysis,
            "direction_summary": direction_summary
        }

        print("\nSaving analysis results...")

        for name, dataframe in outputs.items():

            output_path = OUTPUT_DIR / name

            (
                dataframe.write
                .mode("overwrite")
                .parquet(str(output_path))
            )

            print(f"Saved: {output_path}")

        # Display important findings.
        print("\n========== DISEASE ERROR RANKING ==========")
        disease_errors.show(20, truncate=False)

        print("\n========== TOP 20 DISTRICTS BY WAPE ==========")
        district_errors.show(20, truncate=False)

        print("\n========== ERROR BY FORECAST HORIZON ==========")
        horizon_errors.show(20, truncate=False)

        print("\n========== UNDERPREDICTION / OVERPREDICTION ==========")
        direction_summary.show(20, truncate=False)

        print("\n========== TOP 20 LARGEST PREDICTION ERRORS ==========")
        largest_errors.show(20, truncate=False)

        print("\n========== TOP 20 POTENTIAL SURGE WEEKS ==========")
        weekly_surge_analysis.filter(
            F.col("surge_detected")
        ).show(20, truncate=False)

        print("\n========== ANALYSIS SUMMARY ==========")
        print(f"Model: {MODEL_NAME}")
        print(f"Prediction records: {total_records:,}")
        print(f"Surge threshold: {SURGE_MULTIPLIER}x predicted cases")
        print(f"Output directory: {OUTPUT_DIR}")

        print("\nFORECAST ERROR ANALYSIS COMPLETED SUCCESSFULLY!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
