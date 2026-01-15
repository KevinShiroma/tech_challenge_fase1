import os
import kagglehub
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.preprocessing import LabelEncoder

# Configuração de diretórios
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")

spark = SparkSession.builder.appName("MentalHealth_ETL").master("local[*]").getOrCreate()

print("--- 1. Baixando e Criando Bronze ---")
path = kagglehub.dataset_download("mahdimashayekhi/mental-health")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df_pd = pd.read_csv(os.path.join(path, csv_file))
spark.createDataFrame(df_pd).write.mode("overwrite").parquet(BRONZE_DIR)

print("--- 2. Tratando Silver ---")
df_silver = spark.read.parquet(BRONZE_DIR)
# Normalização de Gênero e Target (0 e 1)
df_silver = df_silver.withColumn('gender', 
    F.when(F.col('gender') == 'Male', 0).when(F.col('gender') == 'Female', 1).otherwise(None)
).withColumn("mental_health_risk", 
    F.when(F.col("mental_health_risk") == "Low", 0).otherwise(1)
).dropna()
df_silver.write.mode("overwrite").parquet(SILVER_DIR)

print("--- 3. Gerando Gold (Features) ---")
df_pd_gold = spark.read.parquet(SILVER_DIR).toPandas()

# Encoding
le = LabelEncoder()
for col in ['mental_health_history', 'seeks_treatment']:
    df_pd_gold[col] = le.fit_transform(df_pd_gold[col])

# One-Hot Encoding
df_final = pd.get_dummies(df_pd_gold, columns=['employment_status', 'work_environment'], prefix='dummy')

spark.createDataFrame(df_final).write.mode("overwrite").parquet(GOLD_DIR)
print("Dados prontos para plotagem em: data/gold")