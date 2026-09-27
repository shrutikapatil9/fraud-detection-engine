import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from xgboost import XGBClassifier

from preprocess import load_and_preprocess_data
from graph_features import generate_synthetic_graph_data, extract_graph_features


def train_fraud_model():

  X_train, X_test, y_train, y_test = load_and_preprocess_data()

 
  print("\n--- Generating Graph Features ---")
  X_train_g = generate_synthetic_graph_data(X_train, seed=42)
  X_train_final = extract_graph_features(X_train_g, is_training=True)

  X_test_g = generate_synthetic_graph_data(X_test, seed=99)
  X_test_final = extract_graph_features(X_test_g, is_training=False)

 
  num_neg = (y_train == 0).sum()
  num_pos = (y_train == 1).sum()
  scale_pos_weight = num_neg / num_pos
  print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

  
  print("\n--- Training Cost-Sensitive XGBoost Model ---")
  model = XGBClassifier(
      n_estimators=150,
      max_depth=6,
      learning_rate=0.05,
      scale_pos_weight=scale_pos_weight,
      eval_metric="logloss",
      random_state=42,
      n_jobs=-1,
  )

  model.fit(X_train_final, y_train)


  print("\n--- Model Evaluation on Test Set ---")
  y_pred_proba = model.predict_proba(X_test_final)[:, 1]
  y_pred = (y_pred_proba >= 0.5).astype(int)


  precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
  pr_auc = auc(recall, precision)
  avg_precision = average_precision_score(y_test, y_pred_proba)

  print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
  print(f"Average Precision Score:       {avg_precision:.4f}")

  print("\nConfusion Matrix:")
  cm = confusion_matrix(y_test, y_pred)
  print(f"TN: {cm[0][0]:<6} | FP: {cm[0][1]}")
  print(f"FN: {cm[1][0]:<6} | TP: {cm[1][1]}")

  print("\nClassification Report:")
  print(classification_report(y_test, y_pred, digits=4))


  os.makedirs("models", exist_ok=True)
  joblib.dump(model, "models/fraud_model.pkl")

 
  feature_names = list(X_train_final.columns)
  joblib.dump(feature_names, "models/feature_names.pkl")

  print("Model and feature catalog saved successfully to models/")


if __name__ == "__main__":
  train_fraud_model()