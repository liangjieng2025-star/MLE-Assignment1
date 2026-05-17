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
    
    # Connect to ALL 4 bronze tables
    attr_path = bronze_features_directory + f"bronze_attributes_{date_suffix}.csv"
    click_path = bronze_features_directory + f"bronze_clickstream_{date_suffix}.csv"
    fin_path = bronze_features_directory + f"bronze_financials_{date_suffix}.csv"
    lms_path = bronze_features_directory + f"bronze_lms_loan_daily_{date_suffix}.csv"  # <-- 4th File
    
    df_attr = spark.read.csv(attr_path, header=True, inferSchema=True)
    df_click = spark.read.csv(click_path, header=True, inferSchema=True)
    df_fin = spark.read.csv(fin_path, header=True, inferSchema=True)
    df_lms = spark.read.csv(lms_path, header=True, inferSchema=True)  # <-- 4th File

    # ---------------------------------------------------------
    # 1. FOUNDATION (Keys & Duplicates)
    # ---------------------------------------------------------
    # Enforce String type and trim hidden spaces
    df_attr = df_attr.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))
    df_click = df_click.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))
    df_fin = df_fin.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))
    df_lms = df_lms.withColumn("Customer_ID", trim(col("Customer_ID").cast(StringType())))

    # Apply strict 10-character filter to ALL tables to drop the 9-char errors
    df_attr = df_attr.filter(length(col("Customer_ID")) == 10)
    df_click = df_click.filter(length(col("Customer_ID")) == 10)
    df_fin = df_fin.filter(length(col("Customer_ID")) == 10)
    df_lms = df_lms.filter(length(col("Customer_ID")) == 10)
    
    # Deduplicate clickstream to prevent Cartesian explosion
    df_click = df_click.dropDuplicates(["Customer_ID"])

    # ---------------------------------------------------------
    # 2. NUMERIC SCRUB & OUTLIER CAPPING
    # ---------------------------------------------------------
    # Scrub Age (numbers only) and cap anything over 100
    df_attr = df_attr.withColumn("Age", F.regexp_replace(col("Age"), "[^0-9]", "").cast(IntegerType()))
    df_attr = df_attr.withColumn("Age", F.when(col("Age") > 100, lit(None)).otherwise(col("Age")))

    # Scrub poisoned Financial columns using DoubleType
    poisoned_numeric_cols = [
        "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", 
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly"
    ]
    for c in poisoned_numeric_cols:
        if c in df_fin.columns:
            df_fin = df_fin.withColumn(c, F.regexp_replace(col(c), "[^0-9.-]", "").cast(DoubleType()))
                  
    # Cap impossible minimums and maximums
    df_fin = df_fin.withColumn("Num_Bank_Accounts", 
                               F.when(col("Num_Bank_Accounts") < 0, 0)
                               .when(col("Num_Bank_Accounts") > 50, lit(None))
                               .otherwise(col("Num_Bank_Accounts")))

    # ---------------------------------------------------------
    # 3. CATEGORICAL CLEANING
    # ---------------------------------------------------------
    df_fin = df_fin.withColumn("Type_of_Loan", 
                               F.when(col("Type_of_Loan").isin("NULL", "Not Specified", "NA", "NM"), "No Loan")
                               .when(col("Type_of_Loan").rlike(","), "Multiple Loans")
                               .otherwise(col("Type_of_Loan")))
    
    df_fin = df_fin.withColumn("Credit_Mix", 
                               F.when(col("Credit_Mix") == "_", "Unknown")
                               .otherwise(col("Credit_Mix")))
    
    df_fin = df_fin.withColumn("Payment_Behaviour", 
                               F.when(col("Payment_Behaviour") == "!@9#%8", "Unknown")
                               .otherwise(col("Payment_Behaviour")))
                               
    df_fin = df_fin.withColumn("Payment_of_Min_Amount", 
                               F.when(col("Payment_of_Min_Amount") == "NM", "Unknown")
                               .otherwise(col("Payment_of_Min_Amount")))

    # ---------------------------------------------------------
    # 4. LMS LEDGER MATH (The Target Engineering)
    # ---------------------------------------------------------
    # Enforce strict DoubleType on LMS financials to prevent math crashes
    lms_financials = ["loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]
    for c in lms_financials:
        if c in df_lms.columns:
            df_lms = df_lms.withColumn(c, col(c).cast(DoubleType()))

    # Ensure snapshot_date is a true DateType for the F.datediff function
    df_lms = df_lms.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # LAB 3 EXACT PROFESSOR LOGIC:
    df_lms = df_lms.withColumn("mob", col("installment_num").cast(IntegerType()))
    
    # (Defensive division: prevents crash if due_amt = 0)
    df_lms = df_lms.withColumn("installments_missed", 
                               F.when(col("due_amt") > 0, F.ceil(col("overdue_amt") / col("due_amt")))
                               .otherwise(0).cast(IntegerType())).fillna(0)
                               
    df_lms = df_lms.withColumn("first_missed_date", 
                               F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
                               .cast(DateType()))
                               
    df_lms = df_lms.withColumn("dpd", 
                               F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date")))
                               .otherwise(0).cast(IntegerType()))

    # ---------------------------------------------------------
    # 5. PANEL DATA JOIN (1-to-Many Master Broadcast)
    # ---------------------------------------------------------
    # Drop snapshot dates from the master profiles so we don't get column collisions.
    # We leave the native snapshot_date inside df_lms to preserve the historical timeline!
    if "snapshot_date" in df_attr.columns: df_attr = df_attr.drop("snapshot_date")
    if "snapshot_date" in df_click.columns: df_click = df_click.drop("snapshot_date")
    if "snapshot_date" in df_fin.columns: df_fin = df_fin.drop("snapshot_date")

    # We join the master profiles ONTO the LMS ledger. 
    # This correctly broadcasts the 1-to-1 profile to every single installment row.
    df_joined = df_lms.join(df_attr, on="Customer_ID", how="outer") \
                      .join(df_click, on="Customer_ID", how="outer") \
                      .join(df_fin, on="Customer_ID", how="outer")
                       
    # Catch missing string values created by outer join gaps
    string_cols = [c for c, t in df_joined.dtypes if t == 'string']
    df_joined = df_joined.fillna("Unknown", subset=string_cols)

    # Note: We NO LONGER overwrite the snapshot_date with F.lit(snapshot_date) at the end, 
    # because that would destroy the historical dates of the installments needed for the ML Target!

    # ---------------------------------------------------------
    # 6. SAVE AND EXPORT
    # ---------------------------------------------------------
    partition_name = "silver_features_" + date_suffix + '.parquet'
    out_filepath = silver_features_directory + partition_name
    df_joined.write.mode("overwrite").parquet(out_filepath)
    
    print('saved to:', out_filepath)
    return df_joined