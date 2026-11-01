
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_PATH = "data/gold/surge_aware_forecast_experiment/predictions"
OUTPUT_PATH = "data/gold/surge_aware_forecast_experiment/national_analysis"

spark = (
    SparkSession.builder
    .appName("EpiScope-National-Surge-Error-Analysis")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nLoading forecast predictions...")

df = spark.read.parquet(INPUT_PATH)

# Aggregate district forecasts into national disease forecasts.
# Each origin, target week, horizon and model remains separate.
national = (
    df
    .groupBy(
        "week_start",
        "target_week",
        "horizon",
        "disease",
        "model"
    )
    .agg(
        F.sum("actual").alias("actual_cases"),
        F.sum("prediction").alias("predicted_cases"),
        F.sum("warning_signal").alias("warning_districts"),
        F.countDistinct("district").alias("districts")
    )
    .withColumn(
        "absolute_error",
        F.abs(F.col("actual_cases") - F.col("predicted_cases"))
    )
    .withColumn(
        "actual_prediction_ratio",
        F.when(
            F.col("predicted_cases") > 0,
            F.col("actual_cases") / F.col("predicted_cases")
        )
    )
    .withColumn(
        "underprediction",
        F.greatest(
            F.col("actual_cases") - F.col("predicted_cases"),
            F.lit(0.0)
        )
    )
)

print("\nCalculating national disease metrics...")

metrics = (
    national
    .groupBy("disease", "model", "horizon")
    .agg(
        F.count("*").alias("forecast_periods"),
        F.sum("actual_cases").alias("total_actual"),
        F.sum("predicted_cases").alias("total_predicted"),
        F.sum("absolute_error").alias("total_absolute_error"),
        F.sum("underprediction").alias("total_underprediction"),
        F.max("actual_prediction_ratio").alias("worst_actual_prediction_ratio"),
        F.max("warning_districts").alias("max_warning_districts")
    )
    .withColumn(
        "national_WAPE_percent",
        100 * F.col("total_absolute_error")
        / F.when(F.col("total_actual") > 0, F.col("total_actual"))
    )
    .orderBy("disease", "horizon", "model")
)

print("\nFinding the largest national underpredictions...")

largest_errors = (
    national
    .filter(F.col("actual_prediction_ratio").isNotNull())
    .orderBy(
        F.col("actual_prediction_ratio").desc(),
        F.col("underprediction").desc()
    )
)

influenza_errors = (
    largest_errors
    .filter(F.col("disease") == "Influenza")
)

print("\nSaving analysis outputs...")

national.write.mode("overwrite").parquet(
    OUTPUT_PATH + "/national_weekly"
)

metrics.write.mode("overwrite").parquet(
    OUTPUT_PATH + "/disease_metrics"
)

largest_errors.limit(100).write.mode("overwrite").parquet(
    OUTPUT_PATH + "/largest_national_errors"
)

influenza_errors.limit(100).write.mode("overwrite").parquet(
    OUTPUT_PATH + "/largest_influenza_errors"
)

print("\n" + "=" * 75)
print("NATIONAL SURGE ERROR ANALYSIS COMPLETED")
print("=" * 75)

print("\nNATIONAL DISEASE METRICS")
metrics.show(40, truncate=False)

print("\nTOP 20 NATIONAL INFLUENZA UNDERPREDICTIONS")
influenza_errors.select(
    "week_start",
    "target_week",
    "horizon",
    "model",
    "actual_cases",
    "predicted_cases",
    "actual_prediction_ratio",
    "warning_districts"
).show(20, truncate=False)

print("\nOutput location:", OUTPUT_PATH)
print("No ground-truth outbreak file was read.")

spark.stop()
