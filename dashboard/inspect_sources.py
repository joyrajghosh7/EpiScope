
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("EpiScopeDashboardSourceInspection")
    .getOrCreate()
)

sources = {
    "WEEKLY GOLD": "data/gold/weekly_district_disease",
    "ORIGINAL OUTBREAK ALERTS": "data/gold/outbreak_detection/alerts",
    "IMPROVED OUTBREAK ALERTS": "data/gold/outbreak_detection_improved/alerts",
    "SURGE-AWARE FORECASTS": "data/gold/surge_aware_forecasts",
}

for name, path in sources.items():
    print("\n" + "=" * 70)
    print(name)
    print("Path:", path)

    try:
        df = spark.read.parquet(path)

        print("\nSCHEMA:")
        df.printSchema()

        print("RECORD COUNT:", df.count())

        print("SAMPLE:")
        df.show(3, truncate=False)

    except Exception as error:
        print("ERROR:", str(error))

spark.stop()

print("\nSOURCE INSPECTION COMPLETED")
