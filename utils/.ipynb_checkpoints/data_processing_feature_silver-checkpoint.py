import os
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, length, trim, lit
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType
from datetime import datetime

def process_silver_features(snapshot_date_str, bronze_features_directory, silver_features_directory, spark):
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_suffix = snapshot_date_str.replace('-','_')
    
    # Connect to bronze tables
    attr_path = bronze_features_directory + f"bronze_attributes_{date_suffix}.csv"
    click_path = bronze_features_directory + f"bronze_clickstream_{date_suffix}.csv"
    fin_path = bronze_features_directory + f"bronze_financials_{date_suffix}.csv"
    
    df_attr = spark.read.csv(attr_path, header=True, inferSchema=True)
    df_click = spark.read.csv(click_path, header=True, inferSchema=True)
    df_fin = spark.read.csv(fin_path, header=True, inferSchema=True)

   -
    # 1. Check keys and duplicates
    # Enforce String type and trim hidden spaces
    df_attr = df_attr.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))
    df_click = df_click.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))
    df_fin = df_fin.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))

    # Apply strict 10-character filter to all tables to drop errors
    df_attr = df_attr.filter(length(col("Customer_ID")) == 10)
    df_click = df_click.filter(length(col("Customer_ID")) == 10)
    df_fin = df_fin.filter(length(col("Customer_ID")) == 10)
    
    # Deduplicate clickstream to prevent Cartesian explosion
    df_click = df_click.dropDuplicates(["Customer_ID"])

    # Check numeric data values and outliers
    # Scrub Age (numbers only) and cap anything over 100
    df_attr = df_attr.withColumn("Age", F.regexp_replace(col("Age"), "[^0-9]", "").cast(IntegerType()))
    df_attr = df_attr.withColumn("Age", F.when(col("Age") > 100, lit(None)).otherwise(col("Age")))

    # Scrub poisoned Financial columns (allow numbers, decimals, negatives)
    poisoned_numeric_cols = [
        "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", 
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly"
    ]
    for c in poisoned_numeric_cols:
        if c in df_fin.columns:  # <-- Defensive check: Only scrub if the column exists!
            df_fin = df_fin.withColumn(c, F.regexp_replace(col(c), "[^0-9.-]", "").cast(DoubleType()))
                  
    # Cap impossible minimums and maximums
    df_fin = df_fin.withColumn("Num_Bank_Accounts", 
                               F.when(col("Num_Bank_Accounts") < 0, 0)
                               .when(col("Num_Bank_Accounts") > 50, lit(None))
                               .otherwise(col("Num_Bank_Accounts")))

    # Check categorical columns
    # Consolidate fake nulls and comma-separated lists
    df_fin = df_fin.withColumn("Type_of_Loan", 
                               F.when(col("Type_of_Loan").isin("NULL", "Not Specified", "NA", "NM"), "No Loan")
                               .when(col("Type_of_Loan").rlike(","), "Multiple Loans")
                               .otherwise(col("Type_of_Loan")))
    
    # Scrub the garbage symbols and fake placeholders
    df_fin = df_fin.withColumn("Credit_Mix", 
                               F.when(col("Credit_Mix") == "_", "Unknown")
                               .otherwise(col("Credit_Mix")))
    
    df_fin = df_fin.withColumn("Payment_Behaviour", 
                               F.when(col("Payment_Behaviour") == "!@9#%8", "Unknown")
                               .otherwise(col("Payment_Behaviour")))
                               
    df_fin = df_fin.withColumn("Payment_of_Min_Amount", 
                               F.when(col("Payment_of_Min_Amount") == "NM", "Unknown")
                               .otherwise(col("Payment_of_Min_Amount")))

    # Join outer 
    # Drop duplicate snapshot dates before joining
    df_attr = df_attr.drop("snapshot_date")
    df_click = df_click.drop("snapshot_date")
    df_fin = df_fin.drop("snapshot_date")

    # Outer Join on clean IDs
    df_joined = df_attr.join(df_click, on="Customer_ID", how="outer") \
                       .join(df_fin, on="Customer_ID", how="outer")
                       
    # Catch missing string values created by Clickstream-only ghost records
    string_cols = [c for c, t in df_joined.dtypes if t == 'string']
    df_joined = df_joined.fillna("Unknown", subset=string_cols)

    # Reattach clean snapshot date
    df_joined = df_joined.withColumn("snapshot_date", F.lit(snapshot_date).cast(DateType()))

    # Save and export
    partition_name = "silver_features_" + date_suffix + '.parquet'
    out_filepath = silver_features_directory + partition_name
    df_joined.write.mode("overwrite").parquet(out_filepath)
    
    print('saved to:', out_filepath)
    return df_joined