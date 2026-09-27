# Real-Time Financial Fraud Detection Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end financial fraud detection system designed for **low-latency transaction scoring and explainable machine learning**.

The project combines **XGBoost classification**, **transaction-network graph features**, **FastAPI real-time inference**, and **SHAP explainability** into a complete fraud detection pipeline with an interactive Streamlit dashboard.

---

## 🚀 Overview

Financial fraud datasets are highly imbalanced, making conventional accuracy-based evaluation unsuitable for assessing fraud detection performance.

This project addresses the problem through:

* Cost-sensitive XGBoost classification
* Precision-Recall AUC (PR-AUC) evaluation
* Transaction network analysis using NetworkX
* Real-time FastAPI inference
* SHAP-based model explainability
* Interactive Streamlit monitoring and analysis

The system is designed to demonstrate how machine learning, graph analytics, API engineering, and explainable AI can be combined into a production-oriented fraud detection workflow.

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────────────┐
                 │ Incoming Transaction Payload │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │      FastAPI REST API       │
                 │     Real-Time Inference     │
                 └──────────────┬──────────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
          ┌──────────────────┐      ┌────────────────────┐
          │ Feature Scaling  │      │ Transaction Graph  │
          │ & Preprocessing  │      │ NetworkX Features  │
          └────────┬─────────┘      └─────────┬──────────┘
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                    ┌──────────────────────┐
                    │ XGBoost Fraud Model  │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │ Fraud Probability + Flag +  │
                 │ Inference Latency           │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ Streamlit Dashboard + SHAP  │
                 │ Model Explainability        │
                 └─────────────────────────────┘
```

---

## ✨ Key Features

### ⚡ Low-Latency REST Inference

Built with **FastAPI** and designed for rapid transaction scoring using pre-loaded model artifacts, feature scalers, and graph-based lookup data.

The target inference latency is **under 50 ms per request** under the project's local benchmark conditions.

### 🕸️ Transaction Graph Features

Uses **NetworkX** to model relationships within transaction data and derive network-based features such as:

* PageRank
* Degree centrality
* Transaction connectivity
* Network-level behavioral patterns

These features provide additional signals beyond traditional transaction-level attributes.

### ⚖️ Imbalanced Learning

Fraud represents a very small proportion of transactions in the dataset.

The model uses XGBoost's `scale_pos_weight` to apply cost-sensitive learning and focuses evaluation on **Precision-Recall AUC (PR-AUC)** rather than relying on accuracy or ROC-AUC alone.

### 🔍 Explainable AI

Uses **SHAP TreeExplainer** to provide feature-level explanations for model predictions.

The dashboard can be used to inspect which transaction characteristics contributed to a fraud prediction, supporting model analysis and auditing.

### 📊 Interactive Dashboard

A Streamlit dashboard provides an interactive interface for:

* Transaction scoring
* Fraud probability visualization
* Prediction results
* Model explanations
* SHAP-based feature attribution

---

## 📁 Project Structure

```text
fraud-detection-engine/
│
├── data/
│   └── creditcard.csv          # Local dataset (gitignored)
│
├── models/
│   ├── trained model files
│   ├── scalers
│   └── graph feature lookups
│
├── src/
│   ├── preprocess.py           # Data preprocessing and feature preparation
│   ├── graph_features.py       # Graph construction and network metrics
│   └── train.py                # Model training and evaluation
│
├── main.py                     # FastAPI application
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** The dataset is intentionally excluded from the repository because `creditcard.csv` exceeds GitHub's individual file-size limit. See the dataset section below for instructions.

---

## 📊 Model Evaluation

Because fraud detection datasets are heavily imbalanced, **Precision-Recall AUC (PR-AUC)** is used as the primary evaluation metric.

| Metric                    | Configuration                 |
| ------------------------- | ----------------------------- |
| Primary Evaluation Metric | Precision-Recall AUC (PR-AUC) |
| Imbalance Handling        | XGBoost `scale_pos_weight`    |
| Approximate Class Weight  | `578.8`                       |
| Target API Latency        | `< 50 ms`                     |
| Explainability            | SHAP TreeExplainer            |
| Graph Analytics           | NetworkX                      |

### Why PR-AUC?

When fraudulent transactions represent a very small percentage of total transactions, a model can achieve a high ROC-AUC while still producing an undesirable number of false positives.

PR-AUC focuses on the relationship between **precision and recall**, making it more informative for highly imbalanced fraud detection tasks.

---

## 🛠️ Tech Stack

| Technology         | Purpose                      |
| ------------------ | ---------------------------- |
| **Python**         | Core development             |
| **XGBoost**        | Fraud classification         |
| **NetworkX**       | Transaction graph analysis   |
| **FastAPI**        | Real-time inference API      |
| **Streamlit**      | Interactive dashboard        |
| **SHAP**           | Model explainability         |
| **scikit-learn**   | Preprocessing and evaluation |
| **Pandas / NumPy** | Data processing              |

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/shrutikapatil9/fraud-detection-engine.git
cd fraud-detection-engine
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

Download the dataset from:

[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

After downloading, place the file at:

```text
data/creditcard.csv
```

The dataset is intentionally **not included in this GitHub repository** because it is larger than GitHub's 100 MB per-file limit.

---

## 🧠 Train the Model

After placing the dataset in the `data/` directory, run:

```bash
python src/train.py
```

The training pipeline performs the required preprocessing, generates graph-based features, trains the XGBoost classifier, evaluates the model using PR-AUC, and saves the required model artifacts.

---

## 🌐 Run the FastAPI Backend

Start the API with:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📊 Run the Streamlit Dashboard

In a separate terminal:

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## 🔄 Example Prediction Flow

A transaction is submitted to the FastAPI endpoint:

```text
Transaction
     │
     ▼
Feature preprocessing
     │
     ▼
Graph feature lookup
     │
     ▼
XGBoost model
     │
     ├── Fraud probability
     ├── Fraud / Legitimate classification
     └── Inference latency
              │
              ▼
       Streamlit Dashboard
              │
              ▼
       SHAP Explanation
```

---

## 🔐 Data & Privacy

The raw transaction dataset is **not committed to this repository**.

The `.gitignore` configuration excludes:

```text
data/creditcard.csv
```

This prevents the large dataset from accidentally being committed to Git.

---

## 📌 Project Goals

This project demonstrates an end-to-end approach to building a machine-learning-powered fraud detection system, including:

* Data preprocessing
* Imbalanced classification
* Graph-based feature engineering
* Model training
* Model evaluation
* Real-time API deployment
* Explainable AI
* Interactive visualization

It is intended as a technical demonstration of combining **machine learning, graph analytics, backend API development, and explainability** in a financial fraud detection use case.

---


