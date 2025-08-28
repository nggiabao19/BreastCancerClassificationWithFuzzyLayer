from pyspark.sql import SparkSession
from pyspark import SparkConf


def init_spark_session(app_name):
    spark = SparkSession.builder \
        .appName("UNetTraining") \
        .master("spark://172.20.201.154:7077") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.instances", "12") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark