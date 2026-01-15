# Databricks notebook source
import pandas as pd

# COMMAND ----------

df_gold = sql('SELECT * FROM kevin_catalog.silver.tb_saude_mental')
display(df_gold.limit(10))

# COMMAND ----------

df_pd = df_gold.toPandas()

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
colunas = ['mental_health_history','seeks_treatment']

label_encoder = LabelEncoder()

for coluna in colunas:
    df_pd[coluna] = label_encoder.fit_transform(df_pd[coluna])

# COMMAND ----------

dummy_employment_status = pd.get_dummies(df_pd['employment_status'], prefix='dummy')
dummy_work_environment = pd.get_dummies(df_pd['work_environment'], prefix='dummy')

dados_dummy = pd.concat([df_pd, dummy_employment_status, dummy_work_environment], axis=1)

# COMMAND ----------

dados_dummy.drop(['employment_status', 'work_environment'], axis=1, inplace=True)

# COMMAND ----------

dados_dummy.display()

# COMMAND ----------

df_gold = spark.createDataFrame(dados_dummy)

# COMMAND ----------

df_gold.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable("kevin_catalog.gold.tb_saude_mental")