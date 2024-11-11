from pyspark.sql import SparkSession
spark = SparkSession.builder().appName("Test").getOrCreate()
test_data = [("Flight1", "2024-01-01", 30), 
             ("Flight2", "2024-01-01", 15)]

df = spark.createDataFrame(test_data,["flight_id","date","delay"])

df.show()
