#%%
import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configuração visual
sb.set_theme(style="whitegrid")

# Carregar dados
spark = SparkSession.builder.master("local[*]").getOrCreate()
df = spark.read.parquet(os.path.join(os.getcwd(), "data", "gold")).toPandas()

# Preparação X e y
X = df.drop(columns=['mental_health_risk'])
y = df['mental_health_risk']

# Split e Escala
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

#%%
# GRÁFICO 1: Correlação (Heatmap)
plt.figure(figsize=(10, 8))
corr = df.corr().round(2)
sb.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de Correlação Geral")
plt.show()

#%%
# GRÁFICO 2: Curva de Erro do KNN (Elbow Method)
error = []
print("Treinando KNNs para curva de erro...")
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_sc, y_train)
    pred_i = knn.predict(X_test_sc)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro x Valor K (KNN)')
plt.xlabel('Valor de K')
plt.ylabel('Erro Médio')
plt.show()

#%%
# GRÁFICO 3: Matriz de Confusão (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred = rf.predict(X_test_sc)

plt.figure(figsize=(6, 5))
sb.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - Random Forest")
plt.ylabel("Real")
plt.xlabel("Predito")
plt.show()

#%%
# GRÁFICO 4: Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='skyblue')
plt.title("Importância das Variáveis (Random Forest)")
plt.xlabel("Importância")
plt.show()