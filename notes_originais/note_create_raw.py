# Databricks notebook source
!pip install kagglehub[pandas-datasets]

# COMMAND ----------

import kagglehub
import os
import pandas as pd

# Baixa o dataset
path = kagglehub.dataset_download("mahdimashayekhi/mental-health")

arquivos = os.listdir(path)

# Buscando o primeiro arquivo .csv da pasta
arquivo_csv = [f for f in arquivos if f.endswith('.csv')][0]
full_path = os.path.join(path, arquivo_csv)

# Criando dataframe pandas o arquivo
df = pd.read_csv(full_path)

# COMMAND ----------

df.describe()

# COMMAND ----------

display(df)

# COMMAND ----------

df_bronze = spark.createDataFrame(df)

# COMMAND ----------

df_bronze.write.mode("overwrite").saveAsTable("kevin_catalog.bronze.tb_saude_mental")