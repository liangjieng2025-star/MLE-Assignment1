import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# FIXED: Added the missing 'col' import
from pyspark.sql.functions import col 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("Pre_ML_Checks").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# FIXED: Let's check August 2024 instead of December to see if we capture the 12-month cohort
gold_path = "datamart/gold/feature_store/gold_feature_store_2024_08_01_90dpd_3mob.parquet"
df_gold = spark.read.parquet(gold_path)

print(f"Loaded Gold Feature Store: {df_gold.count()} Rows\n")

# Prevent the script from crashing if it hits 0 rows again
if df_gold.count() == 0:
    print("CRITICAL: 0 rows found. The mob=12 filter wiped out the dataset for this specific month.")
    print("Action: Go back to main.py, change TARGET_MOB to 6 or 3, and rerun the pipeline.")
    import sys
    sys.exit()

print("=== CHECK 1: ML TYPE COMPATIBILITY ===")
ml_incompatible = [f.name for f in df_gold.schema.fields if isinstance(f.dataType, pyspark.sql.types.StringType)]
print(f"String columns found (Requires One-Hot Encoding before ML): {ml_incompatible}\n")


print("=== CHECK 2: TARGET CLASS IMBALANCE ===")
total_rows = df_gold.count()
imbalance = df_gold.groupBy("label").count().withColumn("percentage", F.round((col("count") / total_rows) * 100, 2))
imbalance.show()


print("=== CHECK 3: NULL LEAKAGE ===")
null_counts = df_gold.select([F.sum(col(c).isNull().cast("int")).alias(c) for c in df_gold.columns]).toPandas()
print("Columns with Nulls (Should be empty):")
print(null_counts.loc[:, (null_counts != 0).any(axis=0)])


print("\n=== CHECK 4: CORRELATION MATRIX (MULTICOLLINEARITY) ===")
numeric_cols = [f.name for f in df_gold.schema.fields if isinstance(f.dataType, (pyspark.sql.types.DoubleType, pyspark.sql.types.IntegerType))]
numeric_cols.remove("label") 

assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
df_vector = assembler.transform(df_gold).select("features")

pearson_corr = Correlation.corr(df_vector, "features").head()[0]
corr_matrix = pearson_corr.toArray()

corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)

print("Highly Correlated Feature Pairs (> 0.8):")
found_correlation = False
for i in range(len(corr_df.columns)):
    for j in range(i):
        if abs(corr_df.iloc[i, j]) > 0.8:
            print(f" - {corr_df.columns[i]} & {corr_df.columns[j]}: {corr_df.iloc[i, j]:.2f}")
            found_correlation = True

if not found_correlation:
    print(" - No high correlations found. Features are clean and independent.")