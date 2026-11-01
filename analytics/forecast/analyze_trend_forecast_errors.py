
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# --------------------------------------------------
# EpiScope: Comparative Forecast Error Analysis
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("EpiScope-Trend-Forecast-Error-Analysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = "data/gold/forecast_trend_experiment/predictions"
OUTPUT_PATH = "data/gold/trend_forecast_error_analysis"

print("\nLoading forecast predictions...")

df = spark.read.parquet(INPUT_PATH)

print("Total prediction records:", df.count())

# --------------------------------------------------
# 1. Calculate forecast errors
# --------------------------------------------------

print("\nCalculating forecast errors...")

df = (
    df
    .withColumn(
        "signed_error",
        F.col("actual_cases") - F.col("prediction")
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
        F.when(F.col("signed_error") > 0, 1).otherwise(0)
    )
    .withColumn(
        "overpredicted",
        F.when(F.col("signed_error") < 0, 1).otherwise(0)
    )
    .withColumn(
        "exact_prediction",
        F.when(F.col("signed_error") == 0, 1).otherwise(0)
    )
    .withColumn(
        "surge_ratio",
        F.when(
            F.col("prediction") > 0,
            F.col("actual_cases") / F.col("prediction")
        )
        .when(
            F.col("actual_cases") > 0,
            F.lit(float("inf"))
        )
        .otherwise(F.lit(1.0))
    )
    .withColumn(
        "is_surge",
        F.when(F.col("surge_ratio") >= 1.5, 1).otherwise(0)
    )
)

# --------------------------------------------------
# 2. Overall model performance
# --------------------------------------------------

print("\nCalculating overall model performance...")

overall = (
    df.groupBy("model", "horizon")
    .agg(
        F.count("*").alias("observations"),
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        (
            F.sum("absolute_error")
            / F.sum(F.abs("actual_cases"))
            * 100
        ).alias("WAPE_percent"),
        F.sum("underpredicted").alias("underpredictions"),
        F.sum("overpredicted").alias("overpredictions"),
        F.sum("exact_prediction").alias("exact_predictions"),
        F.sum("is_surge").alias("surge_observations")
    )
    .orderBy("horizon", "model")
)

# --------------------------------------------------
# 3. Performance by disease
# --------------------------------------------------

print("\nCalculating disease-level errors...")

disease_errors = (
    df.groupBy("model", "disease", "horizon")
    .agg(
        F.count("*").alias("observations"),
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        (
            F.sum("absolute_error")
            / F.sum(F.abs("actual_cases"))
            * 100
        ).alias("WAPE_percent"),
        F.avg("underpredicted").alias("underprediction_rate"),
        F.sum("is_surge").alias("surge_observations")
    )
    .orderBy("disease", "horizon", "model")
)

# --------------------------------------------------
# 4. Performance by district
# --------------------------------------------------

print("\nCalculating district-level errors...")

district_errors = (
    df.groupBy("model", "district")
    .agg(
        F.count("*").alias("observations"),
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        (
            F.sum("absolute_error")
            / F.sum(F.abs("actual_cases"))
            * 100
        ).alias("WAPE_percent")
    )
    .orderBy(
        F.desc("WAPE_percent"),
        "model"
    )
)

# --------------------------------------------------
# 5. Performance by forecast horizon
# --------------------------------------------------

print("\nCalculating horizon-level errors...")

horizon_errors = (
    df.groupBy("model", "horizon")
    .agg(
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        (
            F.sum("absolute_error")
            / F.sum(F.abs("actual_cases"))
            * 100
        ).alias("WAPE_percent"),
        F.avg("underpredicted").alias("underprediction_rate"),
        F.avg("overpredicted").alias("overprediction_rate")
    )
    .orderBy("horizon", "model")
)

# --------------------------------------------------
# 6. Find the largest individual forecast errors
# --------------------------------------------------

print("\nFinding the largest forecast errors...")

largest_errors = (
    df.select(
        "model",
        "district",
        "disease",
        "origin_week",
        "forecast_week",
        "horizon",
        "actual_cases",
        "prediction",
        "signed_error",
        "absolute_error",
        "surge_ratio",
        "is_surge"
    )
    .orderBy(F.desc("absolute_error"))
    .limit(500)
)

# --------------------------------------------------
# 7. Analyze surge periods
# --------------------------------------------------

print("\nAnalyzing surge observations...")

surge_rows = (
    df.filter(F.col("is_surge") == 1)
    .select(
        "model",
        "district",
        "disease",
        "origin_week",
        "forecast_week",
        "horizon",
        "actual_cases",
        "prediction",
        "surge_ratio",
        "absolute_error"
    )
)

surge_summary = (
    df.groupBy("model", "disease", "horizon")
    .agg(
        F.count("*").alias("total_observations"),
        F.sum("is_surge").alias("surge_observations"),
        (
            F.sum("is_surge") / F.count("*") * 100
        ).alias("surge_percentage"),
        F.avg(
            F.when(
                F.col("is_surge") == 1,
                F.col("absolute_error")
            )
        ).alias("average_surge_error"),
        F.avg(
            F.when(
                F.col("is_surge") == 1,
                F.col("surge_ratio")
            )
        ).alias("average_surge_ratio")
    )
    .orderBy("disease", "horizon", "model")
)

# --------------------------------------------------
# 8. Nationwide weekly disease surge analysis
# --------------------------------------------------

print("\nBuilding nationwide weekly analysis...")

weekly_national = (
    df.groupBy(
        "model",
        "disease",
        "forecast_week",
        "horizon"
    )
    .agg(
        F.sum("actual_cases").alias("actual_cases"),
        F.sum("prediction").alias("predicted_cases"),
        F.sum("absolute_error").alias("absolute_error")
    )
    .withColumn(
        "signed_error",
        F.col("actual_cases") - F.col("predicted_cases")
    )
    .withColumn(
        "surge_ratio",
        F.when(
            F.col("predicted_cases") > 0,
            F.col("actual_cases") / F.col("predicted_cases")
        )
        .when(
            F.col("actual_cases") > 0,
            F.lit(float("inf"))
        )
        .otherwise(F.lit(1.0))
    )
    .withColumn(
        "is_surge",
        F.when(F.col("surge_ratio") >= 1.5, 1).otherwise(0)
    )
    .orderBy("disease", "forecast_week", "horizon", "model")
)

# --------------------------------------------------
# 9. Save all analysis outputs
# --------------------------------------------------

print("\nSaving error-analysis outputs...")

outputs = {
    "overall": overall,
    "disease_errors": disease_errors,
    "district_errors": district_errors,
    "horizon_errors": horizon_errors,
    "largest_errors": largest_errors,
    "surge_rows": surge_rows,
    "surge_summary": surge_summary,
    "weekly_national": weekly_national
}

for name, result in outputs.items():
    result.write.mode("overwrite").parquet(
        f"{OUTPUT_PATH}/{name}"
    )

# --------------------------------------------------
# 10. Display important results
# --------------------------------------------------

print("\n========== OVERALL MODEL PERFORMANCE ==========")
overall.show(100, truncate=False)

print("\n========== DISEASE-LEVEL PERFORMANCE ==========")
disease_errors.show(100, truncate=False)

print("\n========== HORIZON PERFORMANCE ==========")
horizon_errors.show(100, truncate=False)

print("\n========== TOP 20 LARGEST FORECAST ERRORS ==========")
largest_errors.show(20, truncate=False)

print("\n========== SURGE PERFORMANCE ==========")
surge_summary.show(100, truncate=False)

print("\n========== NATIONWIDE INFLUENZA SURGE WEEKS ==========")

(
    weekly_national
    .filter(
        (F.col("disease") == "Influenza")
        & (F.col("forecast_week") >= F.lit("2024-11-01"))
        & (F.col("forecast_week") <= F.lit("2025-01-31"))
    )
    .orderBy("forecast_week", "horizon", "model")
    .show(100, truncate=False)
)

print("\n========== EXPERIMENT SUMMARY ==========")
print("Prediction records:", df.count())
print("Models analyzed: 3")
print("Forecast horizons: 4")
print("Surge threshold: Actual cases >= 1.5 times predicted cases")
print("Output folder:", OUTPUT_PATH)

print("\nComparative forecast error analysis completed.")

spark.stop()
