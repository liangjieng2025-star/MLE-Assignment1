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


def process_bronze_features(snapshot_date_str, bronze_features_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    feature_files = {
        "attributes": "data/features_attributes.csv",
        "clickstream": "data/feature_clickstream.csv",
        "financials": "data/features_financials.csv"
    }

    # load data - IRL ingest from back end source system
    for feature_name, filepath in feature_files.items():
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        
        # filter if snapshot_date exists to prevent data leakage
        if 'snapshot_date' in df.columns:
            df = df.filter(col('snapshot_date') == snapshot_date)
            
        print(snapshot_date_str + f' {feature_name} row count:', df.count())
        
        # save bronze table to datamart - IRL connect to database to write
        partition_name = f"bronze_{feature_name}_" + snapshot_date_str.replace('-','_') + '.csv'
        out_filepath = bronze_features_directory + partition_name
        df.toPandas().to_csv(out_filepath, index=False)
        print('saved to:', out_filepath)

    return True