FROM python:3.9-slim

# Instala Java (Obrigatório para o Spark rodar)
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless && \
    apt-get clean

# Define variável do Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

# Instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os scripts
COPY src/ ./src/

# Garante que a pasta de dados exista (vazia)
RUN mkdir -p data

# O comando padrão roda o ETL.
# Para a análise interativa, o usuário usará o VS Code conectado ao container ou rodará localmente.
CMD ["python", "src/01_etl.py"]