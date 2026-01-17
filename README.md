# Tech Challenge - Fase 1: Sistema Inteligente de Suporte ao Diagnóstico

Este repositório contém o projeto da **Fase 1** do Tech Challenge (Pós-Tech Data Engineering). O objetivo é desenvolver um pipeline de dados *end-to-end* e um modelo de Machine Learning para auxiliar no diagnóstico de riscos à saúde mental, utilizando a arquitetura Medalhão.

## Link Para youtube
link https://youtu.be/lHG56KRjrpM

## 📋 Sobre o Projeto

O sistema processa dados médicos para classificar se um paciente possui ou não risco de desenvolver condições de saúde mental. A solução foi adaptada de um ambiente Databricks para uma arquitetura local reprodutível utilizando Docker e Python.

### Arquitetura de Dados
O pipeline segue a arquitetura **Medallion** (Bronze, Silver, Gold):
1.  **Bronze (Raw):** Ingestão dos dados brutos diretamente da API do Kaggle.
2.  **Silver (Cleaned):** Limpeza de nulos, normalização de colunas categóricas e conversão de tipos (Spark).
3.  **Gold (Curated):** Engenharia de features (*One-Hot Encoding*, *Label Encoding*) pronta para o consumo do modelo de ML.

### Tecnologias Utilizadas
* **Linguagem:** Python 3.9
* **Processamento de Dados:** PySpark & Pandas
* **Machine Learning:** Scikit-Learn (KNN e Random Forest)
* **Visualização:** Matplotlib & Seaborn
* **Ambiente:** Docker & VS Code
* **Fonte de Dados:** [Kaggle - Mental Health Dataset](https://www.kaggle.com/datasets/mahdimashayekhi/mental-health)

---

## 📂 Estrutura do Repositório

```text
tech-challenge-fase1/
│
├── Dockerfile                  # Receita para criar o ambiente (Python + Java/Spark)
├── requirements.txt            # Lista de bibliotecas necessárias
├── README.md                   # Documentação do projeto
├── .gitignore                  # Arquivos ignorados (dados e caches)
│
├── src/                        # Código-fonte
│   ├── 01_etl.py               # Script de ETL: Baixa do Kaggle e processa até a camada Gold
│   └── 02_analise_interativa.py # Script para geração de gráficos e modelos (Janela Interativa)
│
└── data/                       # [GitIgnored] Pasta local onde os dados processados serão salvos
    ├── bronze/
    ├── silver/
    └── gold/
```

## 🚀 Guia de Execução

Para garantir que o projeto rode em qualquer máquina sem conflitos de dependência, utilizamos Docker para a preparação dos dados pesados.

### Pré-requisitos
* Docker Desktop instalado e rodando.
* VS Code com a extensão Python instalada.

### Passo 1: Construir a Imagem Docker
Abra o terminal na raiz do projeto e execute o comando abaixo para criar a imagem com Spark e Python configurados:

```bash
docker build -t tech-challenge-health .
```

### Passo 2: Executar o Pipeline ETL (Ingestão e Tratamento)
Este comando iniciará um container que baixa os dados do Kaggle, processa as camadas Bronze/Silver e salva a camada Gold (Parquet) na sua pasta local `data/`.

**No Linux/Mac:**
```bash
docker run --rm -v $(pwd)/data:/app/data tech-challenge-health
```

**No Windows (PowerShell):**
```powershell
docker run --rm -v ${PWD}/data:/app/data tech-challenge-health
```

> **Atenção:** O download inicial do dataset e dependências do Java pode levar alguns minutos. Aguarde até aparecer a mensagem no terminal: *"Dados prontos para plotagem em: data/gold"*.

### Passo 3: Análise e Geração de Gráficos (Local)
Após a execução do Passo 2, a pasta `data/gold` estará populada com os dados tratados. Agora, você deve rodar a análise interativa no VS Code para visualizar os gráficos.

1. Instale as dependências locais (opcional, se não estiver usando Dev Container):
   ```bash
   pip install -r requirements.txt
   ```
2. Abra o arquivo `src/02_analise_interativa.py` no VS Code.
3. No código, você verá células marcadas com `#%%`.
4. Clique na opção **"Run Cell"** (ou "Executar Célula") que aparece logo acima de cada bloco de código.
5. Os gráficos interativos aparecerão na janela lateral (*Interactive Window*).

---

## ✒️ Autor - Kevin Makoto Shiroma
Projeto desenvolvido como parte da avaliação do **Tech Challenge - Fase 1**.
