import os
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType
from datetime import datetime

def process_gold_features(snapshot_date_str, silver_features_directory, gold_feature_store_directory, spark, dpd_threshold=90, mob_target=12):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_features_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Label engineering
    # Filter down to the specific Month on Book (mob) we want to predict
    df = df.filter(col("mob") == mob_target)

    # Create the binary target label based on the Days Past Due (dpd) threshold
    df = df.withColumn("label", F.when(col("dpd") >= dpd_threshold, 1).otherwise(0).cast(IntegerType()))
    
    # Add metadata so we know exactly what this target represents
    label_def_str = f"{dpd_threshold}dpd_{mob_target}mob"
    df = df.withColumn("label_def", F.lit(label_def_str).cast(StringType()))

    # Data Leakage Check
    # Drop PII and raw strings
    columns_to_drop = ["Name", "SSN"]
    for c in columns_to_drop:
        if c in df.columns:
            df = df.drop(c)
   
    leakage_cols = ["installments_missed", "first_missed_date", "dpd", "overdue_amt", "due_amt"]
    for c in leakage_cols:
        if c in df.columns:
            df = df.drop(c)

    # Fill remaining nulls
    df = df.fillna(0)

    # Dynamically name the file based on the target (e.g., gold_feature_store_2023_01_01_90dpd_12mob.parquet)
    out_partition = "gold_feature_store_" + snapshot_date_str.replace('-','_') + f"_{label_def_str}.parquet"
    out_filepath = gold_feature_store_directory + out_partition
    
    df.write.mode("overwrite").parquet(out_filepath)
    print('saved Gold feature store to:', out_filepath)

    return df