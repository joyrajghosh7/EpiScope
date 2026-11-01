
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# --------------------------------------------------
# EpiScope: Trend-Aware Forecasting Experiment
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("EpiScope-Trend-Forecast-Experiment")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = "data/gold/weekly_district_disease"
OUTPUT_PATH = "data/gold/forecast_trend_experiment"

print("\nLoading weekly Gold data...")

df = spark.read.parquet(INPUT_PATH)

required = {"district", "disease", "week_start", "total_cases"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = (
    df.select(
        "district",
        "disease",
        F.to_date("week_start").alias("week_start"),
        F.col("total_cases").cast("double").alias("actual_cases")
    )
    .filter(F.col("disease") != "Other")
)

# One row per district, disease and week is expected.
duplicates = (
    df.groupBy("district", "disease", "week_start")
    .count()
    .filter(F.col("count") > 1)
    .limit(1)
    .count()
)

if duplicates:
    raise ValueError(
        "Duplicate district-disease-week records found in Gold data."
    )

series_window = (
    Window.partitionBy("district", "disease")
    .orderBy("week_start")
)

history_window_12 = series_window.rowsBetween(-12, -1)
recent_4_window = series_window.rowsBetween(-4, -1)
previous_4_window = series_window.rowsBetween(-8, -5)

print("Building historical trend features...")

features = (
    df
    .withColumn(
        "history_weeks",
        F.count("actual_cases").over(
            series_window.rowsBetween(Window.unboundedPreceding, -1)
        )
    )
    .withColumn(
        "recent12",
        F.avg("actual_cases").over(history_window_12)
    )
    .withColumn(
        "recent4",
        F.avg("actual_cases").over(recent_4_window)
    )
    .withColumn(
        "previous4",
        F.avg("actual_cases").over(previous_4_window)
    )
)

# Generate four forecast horizons for every eligible origin.
horizons = spark.createDataFrame(
    [(1,), (2,), (3,), (4,)],
    ["horizon"]
)

origins = (
    features
    .filter(
        (F.col("history_weeks") >= 52)
        & F.col("recent12").isNotNull()
        & F.col("recent4").isNotNull()
        & F.col("previous4").isNotNull()
    )
    .select(
        "district",
        "disease",
        F.col("week_start").alias("origin_week"),
        "recent12",
        "recent4",
        "previous4"
    )
)

forecast_rows = (
    origins
    .crossJoin(horizons)
    .withColumn(
    "forecast_week",
    F.date_add(
        "origin_week",
        (F.col("horizon") * 7).cast("int")
    )
)
)

# Actual value for each forecast target week.
actuals = df.select(
    "district",
    "disease",
    F.col("week_start").alias("forecast_week"),
    F.col("actual_cases")
)

# Seasonal value: same forecast week, 52 weeks earlier.
seasonal = df.select(
    "district",
    "disease",
    F.date_add("week_start", 364).alias("forecast_week"),
    F.col("actual_cases").alias("seasonal_cases")
)

print("Joining actual and seasonal values...")

evaluation = (
    forecast_rows
    .join(
        actuals,
        ["district", "disease", "forecast_week"],
        "inner"
    )
    .join(
        seasonal,
        ["district", "disease", "forecast_week"],
        "left"
    )
    .filter(F.col("seasonal_cases").isNotNull())
)

# Guard against zero previous activity and limit extreme ratios.
trend_ratio = (
    F.when(
        F.col("previous4") > 0,
        F.col("recent4") / F.col("previous4")
    )
    .otherwise(F.lit(1.0))
)

evaluation = (
    evaluation
    .withColumn(
        "trend_factor",
        F.least(
            F.lit(2.0),
            F.greatest(F.lit(0.5), trend_ratio)
        )
    )
    .withColumn(
        "trend_adjusted_recent",
        F.col("recent12") * F.col("trend_factor")
    )
)

# Three forecasting models.
evaluation = (
    evaluation
    .withColumn(
        "recent_mean_prediction",
        F.col("recent12")
    )
    .withColumn(
        "seasonal_recent_prediction",
        0.60 * F.col("recent12")
        + 0.40 * F.col("seasonal_cases")
    )
    .withColumn(
        "trend_seasonal_prediction",
        0.40 * F.col("recent12")
        + 0.20 * F.col("trend_adjusted_recent")
        + 0.40 * F.col("seasonal_cases")
    )
)

# Convert the three predictions into a common model column.
models = (
    evaluation
    .select(
        "district",
        "disease",
        "origin_week",
        "forecast_week",
        "horizon",
        "actual_cases",
        F.explode(
            F.array(
                F.struct(
                    F.lit("Recent Mean").alias("model"),
                    F.col("recent_mean_prediction").alias("prediction")
                ),
                F.struct(
                    F.lit("Seasonal-Recent Mean").alias("model"),
                    F.col("seasonal_recent_prediction").alias("prediction")
                ),
                F.struct(
                    F.lit("Trend-Seasonal Mean").alias("model"),
                    F.col("trend_seasonal_prediction").alias("prediction")
                )
            )
        ).alias("result")
    )
    .select(
        "district",
        "disease",
        "origin_week",
        "forecast_week",
        "horizon",
        "actual_cases",
        F.col("result.model").alias("model"),
        F.col("result.prediction").alias("prediction")
    )
    .withColumn(
        "absolute_error",
        F.abs(F.col("actual_cases") - F.col("prediction"))
    )
    .withColumn(
        "squared_error",
        F.pow(F.col("actual_cases") - F.col("prediction"), 2)
    )
    .withColumn(
        "absolute_actual",
        F.abs(F.col("actual_cases"))
    )
)

print("Calculating overall metrics...")

def calculate_metrics(data, group_columns):
    return (
        data.groupBy(*group_columns)
        .agg(
            F.count("*").alias("observations"),
            F.avg("absolute_error").alias("MAE"),
            F.sqrt(F.avg("squared_error")).alias("RMSE"),
            (
                F.sum("absolute_error")
                / F.sum("absolute_actual")
                * 100
            ).alias("WAPE_percent")
        )
        .orderBy(*group_columns)
    )

overall = calculate_metrics(
    models,
    ["model", "horizon"]
)

by_disease = calculate_metrics(
    models,
    ["model", "horizon", "disease"]
)

# Save detailed predictions and metrics separately.
print("Saving experiment outputs...")

(
    models.write
    .mode("overwrite")
    .partitionBy("model", "horizon")
    .parquet(f"{OUTPUT_PATH}/predictions")
)

(
    overall.write
    .mode("overwrite")
    .parquet(f"{OUTPUT_PATH}/metrics/overall")
)

(
    by_disease.write
    .mode("overwrite")
    .parquet(f"{OUTPUT_PATH}/metrics/by_disease")
)

print("\n========== OVERALL FORECAST METRICS ==========")
overall.show(100, truncate=False)

print("\n========== METRICS BY DISEASE ==========")
by_disease.show(100, truncate=False)

print("\n========== EXPERIMENT SUMMARY ==========")
print("Evaluation prediction records:", models.count())
print("Models tested: 3")
print("Forecast horizons: 4 weeks")
print("Output folder:", OUTPUT_PATH)

print("\nTrend-aware forecasting experiment completed.")

spark.stop()
