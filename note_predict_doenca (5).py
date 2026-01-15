# Databricks notebook source
# MAGIC %md
# MAGIC # Tech Challenge - Fase 1: Sistema Inteligente de Suporte ao Diagnóstico
# MAGIC Este projeto visa implementar a base de um sistema de IA para um hospital universitário, focado em auxiliar a equipe clínica na análise inicial de dados médicos. O objetivo é realizar a triagem automática de risco de saúde mental utilizando algoritmos de Machine Learning.
# MAGIC
# MAGIC **Tarefa Principal:** Classificação binária ("tem ou não tem risco") com base em dados estruturados.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregando os dados da tabela gold

# COMMAND ----------

df_spark = sql('SELECT * FROM kevin_catalog.gold.tb_saude_mental')
display(df_spark.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploração e Limpeza de Dados
# MAGIC Nesta fase, carregamos os dados em Pandas para análise estatística e visualização.

# COMMAND ----------

# Conversão para Pandas para facilitar a análise exploratória
df_pd = df_spark.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importanto bib para verificar dados nulos

# COMMAND ----------

!pip install missingno

# COMMAND ----------

import missingno as msno

# Visualização de dados ausentes
msno.matrix(df_pd)
df_pd.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise de Distribuição e Correlação
# MAGIC Verificamos como os dados estão distribuídos e como as variáveis se relacionam.

# COMMAND ----------

import matplotlib.pyplot as plt

# Matriz de correlação para identificar variáveis que impactam o diagnóstico
corr_matriz = df_pd.corr().round(2)
fog, ax = plt.subplots(figsize=(8,8))
sb.heatmap(data=corr_matriz, annot=True, linewidths=5, ax=ax)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pré-processamento
# MAGIC Preparamos os dados separando as características (X) do alvo (y) e realizando o escalonamento.

# COMMAND ----------

from sklearn.model_selection import *
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Seleção das variáveis preditoras
x = df_pd[
    [
        "age",
        "gender",
        "mental_health_history",
        "seeks_treatment",
        "stress_level",
        "sleep_hours",
        "physical_activity_days",
        "depression_score",
        "anxiety_score",
        "social_support_score",
        "productivity_score",
        "dummy_Employed", 
        "dummy_Self-employed", 
        "dummy_Student", 
        "dummy_Unemployed",
        "dummy_Hybrid", 
        "dummy_On-site", 
        "dummy_Remote"
    ]
]

# COMMAND ----------

# Seleção da variável alvo (diagnóstico)
y = df_pd['mental_health_risk']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Separação de base de treinamento e teste

# COMMAND ----------

# tamanho da base de teste = 20%
# random_state = 42. 
# Distribuição aleatória dos dados para garantir que a distribuição de classes seja a mesma em ambos os conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y, random_state=42)


# COMMAND ----------

# Padronização
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## KNN

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# COMMAND ----------

# Testando diferentes valores de K para encontrar o menor erro médio
error = []

for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i) # n_neighbors = 5 é o número de vizinhos mais próximos que serão considerados para fazer a predição 
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test_escalonado)
    error.append(np.mean(pred_i != y_test))

# COMMAND ----------

plt.figure(figsize=(12,6))
plt.plot(range(1,10), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)

plt.title("Erro médio para K")
plt.xlabel("Valor de K")
plt.ylabel("Erro Médio")

# COMMAND ----------

# Treinamento final com K=6
modelo_class = KNeighborsClassifier(n_neighbors=6)
modelo_class.fit(x_train_escalonado, y_train)
y_predict = modelo_class.predict(x_test_escalonado)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Avaliação e Interpretação

# COMMAND ----------

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_predict))

# COMMAND ----------

print("Relatório de Classificação:")
print(classification_report(y_test, y_predict))

# COMMAND ----------

sb.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d')

# COMMAND ----------

# MAGIC %md
# MAGIC ##  RANDOM FOREST

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

# COMMAND ----------

# Instanciando o modelo (usando 100 árvores de decisão) 
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# COMMAND ----------

# Treinamento do modelo com os dados escalonado
modelo_rf.fit(x_train_escalonado, y_train)

# COMMAND ----------

# Predição com os dados de teste
y_predict_rf = modelo_rf.predict(x_test_escalonado)

# COMMAND ----------

# Avaliação do desempenho
print("--- Relatório de Classificação: Random Forest ---")
print(classification_report(y_test, y_predict_rf))

# COMMAND ----------

# 5. Visualização da Matriz de Confusão
plt.figure(figsize=(6,4))
sb.heatmap(confusion_matrix(y_test, y_predict_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# COMMAND ----------


# Interpretação: Feature Importance (Importância das Variáveis) 
# Este gráfico mostra quais dados médicos mais impactaram a decisão do modelo [cite: 17, 43]
importancias = pd.Series(modelo_rf.feature_importances_, index=x.columns)
importancias_ordenadas = importancias.sort_values(ascending=True)

plt.figure(figsize=(10,6))
importancias_ordenadas.plot(kind='barh', color='skyblue')
plt.title("Importância das Variáveis - Suporte ao Diagnóstico")
plt.xlabel("Nível de Importância")
plt.show()