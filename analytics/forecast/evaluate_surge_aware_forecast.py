
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

INPUT_PATH = "data/gold/weekly_district_disease"
OUTPUT_PATH = "data/gold/surge_aware_forecast_experiment"

spark = (
    SparkSession.builder
    .appName("EpiScope-Surge-Aware-Forecast-Experiment")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nLoading weekly Gold data...")

df = (
    spark.read.parquet(INPUT_PATH)
    .filter(F.col("disease") != "Other")
    .select(
        "district",
        "disease",
        F.to_date("week_start").alias("week_start"),
        F.col("total_cases").cast("double").alias("actual_cases")
    )
    .filter(F.col("week_start").isNotNull())
)

# Each district-disease combination has a complete weekly calendar.
history_window = (
    Window.partitionBy("district", "disease")
    .orderBy("week_start")
)

last_12 = history_window.rowsBetween(-11, 0)
last_4 = history_window.rowsBetween(-3, 0)
previous_4 = history_window.rowsBetween(-7, -4)
previous_8 = history_window.rowsBetween(-11, -4)

print("Calculating historical features...")

features = (
    df
    .withColumn("history_weeks", F.count("actual_cases").over(
        history_window.rowsBetween(Window.unboundedPreceding, 0)
    ))
    .withColumn("recent12", F.avg("actual_cases").over(last_12))
    .withColumn("recent4", F.avg("actual_cases").over(last_4))
    .withColumn("previous4", F.avg("actual_cases").over(previous_4))
    .withColumn("previous8_avg", F.avg("actual_cases").over(previous_8))
    .withColumn("previous8_std", F.stddev_pop("actual_cases").over(previous_8))
    .withColumn("lag52", F.lag("actual_cases", 52).over(history_window))
)

# Use only forecast origins with at least 52 weeks of prior history.
origins = (
    features
    .filter(F.col("history_weeks") >= 52)
    .withColumn(
        "trend_factor",
        F.when(
            F.col("previous4").isNull() | (F.col("previous4") <= 0),
            F.lit(1.0)
        ).otherwise(
            F.least(
                F.lit(2.0),
                F.greatest(
                    F.lit(0.5),
                    F.col("recent4") / F.col("previous4")
                )
            )
        )
    )
    .withColumn(
        "seasonal_value",
        F.coalesce(F.col("lag52"), F.col("recent12"))
    )
    .withColumn(
        "seasonal_recent",
        0.6 * F.col("recent12") + 0.4 * F.col("seasonal_value")
    )
    .withColumn(
        "trend_seasonal",
        0.4 * F.col("recent12")
        + 0.2 * (F.col("recent12") * F.col("trend_factor"))
        + 0.4 * F.col("seasonal_value")
    )
    .withColumn(
        "warning_threshold",
        F.col("previous8_avg") + 2.0 * F.coalesce(
            F.col("previous8_std"), F.lit(0.0)
        )
    )
    .withColumn(
        "warning_signal",
        F.when(
            (F.col("recent4") > F.col("warning_threshold"))
            & (F.col("recent4") > F.col("previous8_avg"))
            & (F.col("previous8_avg").isNotNull()),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    .withColumn(
        "warning_excess",
        F.when(
            F.col("warning_signal") == 1,
            F.greatest(
                F.col("recent4") - F.col("warning_threshold"),
                F.lit(0.0)
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "surge_aware",
        F.least(
            2.0 * F.greatest(F.col("trend_seasonal"), F.lit(0.0)),
            F.greatest(
                F.lit(0.0),
                F.col("trend_seasonal") + 0.5 * F.col("warning_excess")
            )
        )
    )
    .filter(
        F.col("recent12").isNotNull()
        & F.col("recent4").isNotNull()
        & F.col("previous4").isNotNull()
    )
)

print("Creating four forecast horizons...")

horizons = F.array(
    F.lit(1),
    F.lit(2),
    F.lit(3),
    F.lit(4)
)

forecast_origins = (
    origins
    .withColumn("horizon", F.explode(horizons))
    .withColumn(
        "target_week",
        F.date_add(
            F.col("week_start"),
            (F.col("horizon") * 7).cast("int")
        )
    )
)

# Actual future values are joined only for evaluation.
actuals = df.select(
    "district",
    "disease",
    F.col("week_start").alias("target_week"),
    F.col("actual_cases").alias("actual")
)

evaluated = (
    forecast_origins
    .join(
        actuals,
        on=["district", "disease", "target_week"],
        how="inner"
    )
)

print("Preparing model predictions...")

model_predictions = evaluated.select(
    "district",
    "disease",
    "week_start",
    "target_week",
    "horizon",
    "actual",
    "warning_signal",
    F.explode(
        F.array(
            F.struct(
                F.lit("Recent Mean").alias("model"),
                F.col("recent12").alias("prediction")
            ),
            F.struct(
                F.lit("Seasonal-Recent Mean").alias("model"),
                F.col("seasonal_recent").alias("prediction")
            ),
            F.struct(
                F.lit("Trend-Seasonal Mean").alias("model"),
                F.col("trend_seasonal").alias("prediction")
            ),
            F.struct(
                F.lit("Surge-Aware Mean").alias("model"),
                F.col("surge_aware").alias("prediction")
            )
        )
    ).alias("model_result")
).select(
    "district",
    "disease",
    "week_start",
    "target_week",
    "horizon",
    "actual",
    "warning_signal",
    F.col("model_result.model").alias("model"),
    F.greatest(
        F.col("model_result.prediction"),
        F.lit(0.0)
    ).alias("prediction")
)

model_predictions = (
    model_predictions
    .withColumn("error", F.col("actual") - F.col("prediction"))
    .withColumn("absolute_error", F.abs("error"))
    .withColumn("squared_error", F.pow("error", 2))
)

print("Calculating overall metrics...")

def calculate_metrics(data, group_columns):
    return (
        data
        .groupBy(*group_columns)
        .agg(
            F.count("*").alias("observations"),
            F.avg("absolute_error").alias("MAE"),
            F.sqrt(F.avg("squared_error")).alias("RMSE"),
            (
                F.sum("absolute_error")
                / F.when(F.sum("actual") != 0, F.sum("actual"))
            ).alias("WAPE"),
            F.sum(
                F.when(F.col("error") > 0, 1).otherwise(0)
            ).alias("underpredictions"),
            F.sum(
                F.when(F.col("error") < 0, 1).otherwise(0)
            ).alias("overpredictions"),
            F.sum("warning_signal").alias("warning_observations")
        )
        .withColumn("WAPE_percent", F.col("WAPE") * 100)
        .drop("WAPE")
        .orderBy(*group_columns)
    )

overall_metrics = calculate_metrics(
    model_predictions,
    ["model", "horizon"]
)

disease_metrics = calculate_metrics(
    model_predictions,
    ["disease", "model", "horizon"]
)

print("\nSaving prediction results...")

(
    model_predictions
    .write
    .mode("overwrite")
    .partitionBy("model")
    .parquet(OUTPUT_PATH + "/predictions")
)

overall_metrics.write.mode("overwrite").parquet(
    OUTPUT_PATH + "/metrics/overall"
)

disease_metrics.write.mode("overwrite").parquet(
    OUTPUT_PATH + "/metrics/by_disease"
)

print("\n" + "=" * 75)
print("SURGE-AWARE FORECAST EXPERIMENT COMPLETED")
print("=" * 75)

print("\nOVERALL MODEL COMPARISON")
overall_metrics.show(40, truncate=False)

print("\nWEEK 1 PERFORMANCE BY DISEASE")
(
    disease_metrics
    .filter(F.col("horizon") == 1)
    .orderBy("disease", "WAPE_percent")
    .show(40, truncate=False)
)

print("\nOutput location:", OUTPUT_PATH)
print("Ground-truth outbreak file was not used.")

spark.stop()
