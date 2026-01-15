# Databricks notebook source
df_spark = sql('SELECT * FROM kevin_catalog.bronze.tb_saude_mental')

# COMMAND ----------

display(df_spark.limit(10))

# COMMAND ----------

from pyspark.sql import functions as F

df_silver = df_spark.withColumn('gender', 
    F.when(F.col('gender') == 'Male', 0)
     .when(F.col('gender') == 'Female', 1)
     .otherwise(None)
)

df_silver = df_silver.withColumn("mental_health_risk", F.when(F.col("mental_health_risk") == "Low", 0).otherwise(1))

display(df_silver)

# COMMAND ----------

df_silver = df_silver.dropna()
display(df_silver.limit(3))

# COMMAND ----------

df_silver.write.mode("overwrite").saveAsTable("kevin_catalog.silver.tb_saude_mental")