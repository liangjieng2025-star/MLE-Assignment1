import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
# Removed FloatType to enforce strict DoubleType architecture
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType

import utils.data_processing_feature_bronze
import utils.data_processing_feature_silver
import utils.data_processing_feature_gold


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []
    current_date = start_date
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        # safely add exactly one month, automatically handling December rollovers
        current_date += relativedelta(months=1)
    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Dates to process:", dates_str_lst)

# ---------------------------------------------------------
# 1. BRONZE LAYER
# ---------------------------------------------------------
bronze_features_directory = "datamart/bronze/features/"
if not os.path.exists(bronze_features_directory):
    os.makedirs(bronze_features_directory)

print("\n=== Running Bronze Layer Backfill ===")
for date_str in dates_str_lst:
    utils.data_processing_feature_bronze.process_bronze_features(date_str, bronze_features_directory, spark)


# ---------------------------------------------------------
# 2. SILVER LAYER
# ---------------------------------------------------------
silver_features_directory = "datamart/silver/features/"
if not os.path.exists(silver_features_directory):
    os.makedirs(silver_features_directory)

print("\n=== Running Silver Layer Backfill ===")
for date_str in dates_str_lst:
    utils.data_processing_feature_silver.process_silver_features(date_str, bronze_features_directory, silver_features_directory, spark)


# ---------------------------------------------------------
# 3. GOLD LAYER
# ---------------------------------------------------------
gold_feature_store_directory = "datamart/gold/feature_store/"
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

print("\n=== Running Gold Layer Backfill (Credit Risk Default Model) ===")
# Define our strict Default parameters for the Lab 3 objective
TARGET_DPD = 90
TARGET_MOB = 3

for date_str in dates_str_lst:
    # Explicitly pass the dpd_threshold and mob_target to calculate the Label
    utils.data_processing_feature_gold.process_gold_features(
        snapshot_date_str=date_str, 
        silver_features_directory=silver_features_directory, 
        gold_feature_store_directory=gold_feature_store_directory, 
        spark=spark,
        dpd_threshold=TARGET_DPD, 
        mob_target=TARGET_MOB
    )
    
print("\n✅ Full ETL Pipeline Execution Complete!")