from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("EpiScope-Test")
    .master("local[*]")
    .getOrCreate()
)

data = [
    ("Dhaka", "Dengue", 120),
    ("Chattogram", "Dengue", 85),
    ("Sylhet", "Malaria", 40),
    ("Khulna", "Diarrhoea", 65)
]

columns = ["district", "disease", "cases"]

df = spark.createDataFrame(data, columns)

print("\n===== EPISCOPE SPARK TEST =====")
df.show()

print("Total cases:", df.agg({"cases": "sum"}).collect()[0][0])

spark.stop()
print("\nSpark test completed successfully!")