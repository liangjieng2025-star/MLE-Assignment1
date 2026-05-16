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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_gold_features(snapshot_date_str, silver_features_directory, gold_feature_store_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_features_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean up for ML compatibility 
    # dropping columns that cannot be used by the model easily (like raw strings/names)
    columns_to_drop = ["Name", "SSN"]
    for c in columns_to_drop:
        if c in df.columns:
            df = df.drop(c)

    # fill nulls so the model doesn't crash later
    df = df.fillna(0)

    # save gold table - IRL connect to database to write
    out_partition = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    out_filepath = gold_feature_store_directory + out_partition
    
    df.write.mode("overwrite").parquet(out_filepath)
    print('saved to:', out_filepath)

    return df