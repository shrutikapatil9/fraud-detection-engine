import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    data_path: str = "data/creditcard.csv", test_size: float = 0.2
):

  if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found at '{data_path}'. Please place 'creditcard.csv' in"
        " the data/ directory."
    )

  print("Loading dataset...")
  df = pd.read_csv(data_path)
  print(f"Dataset shape: {df.shape}")

  
  X = df.drop(columns=["Class"])
  y = df["Class"]


  scaler_amount = StandardScaler()
  scaler_time = StandardScaler()

  X["scaled_amount"] = scaler_amount.fit_transform(X[["Amount"]])
  X["scaled_time"] = scaler_time.fit_transform(X[["Time"]])

  
  X = X.drop(columns=["Amount", "Time"])

 
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=42, stratify=y
  )

  print(
      f"Train set: {X_train.shape[0]} samples | Fraud ratio:"
      f" {y_train.mean():.4%}"
  )
  print(
      f"Test set:  {X_test.shape[0]} samples  | Fraud ratio:"
      f" {y_test.mean():.4%}"
  )


  os.makedirs("models", exist_ok=True)


  joblib.dump(scaler_amount, "models/scaler_amount.pkl")
  joblib.dump(scaler_time, "models/scaler_time.pkl")
  print("Scalers saved to models/")

  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  load_and_preprocess_data()