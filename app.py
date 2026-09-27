import joblib
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st


st.set_page_config(
    page_title="Fraud Detection Engine & Explainability",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Real-Time Fraud Detection Engine & Model Explainability")
st.markdown(
    "Interactive dashboard for streaming transactions to the REST API and"
    " rendering SHAP decision triggers."
)


@st.cache_resource
def load_artifacts():
  model = joblib.load("models/fraud_model.pkl")
  scaler_amount = joblib.load("models/scaler_amount.pkl")
  scaler_time = joblib.load("models/scaler_time.pkl")
  graph_lookup = joblib.load("models/graph_lookup.pkl")
  feature_names = joblib.load("models/feature_names.pkl")
  explainer = shap.TreeExplainer(model)
  return (
      model,
      scaler_amount,
      scaler_time,
      graph_lookup,
      feature_names,
      explainer,
  )


try:
  model, scaler_amount, scaler_time, graph_lookup, feature_names, explainer = (
      load_artifacts()
  )
except Exception as e:
  st.error(
      f"Failed to load model artifacts: {e}. Please ensure Step 4 is complete."
  )
  st.stop()


st.sidebar.header("Microservice Endpoint")
api_url = st.sidebar.text_input(
    "API Endpoint URL", value="http://127.0.0.1:8000/predict"
)


st.sidebar.header("Transaction Inputs")
amount = st.sidebar.number_input(
    "Transaction Amount ($)", value=250.00, step=10.0
)
time_sec = st.sidebar.number_input("Time Elapsed (s)", value=400.0, step=10.0)
card_id = st.sidebar.text_input("Card ID", value="card_1200")
merchant_id = st.sidebar.text_input("Merchant ID", value="merchant_450")


is_suspicious = st.sidebar.checkbox("Simulate High-Risk Transaction")


pca_features = {}
for i in range(1, 29):
  col = f"V{i}"
  if is_suspicious and col in ["V14", "V10", "V12", "V17"]:
    pca_features[col] = -5.5 
  else:
    pca_features[col] = round(float(np.random.normal(0, 0.5)), 4)


if st.button("Score Transaction via API", type="primary"):
  payload = {
      "time": time_sec,
      "amount": amount,
      "card_id": card_id,
      "merchant_id": merchant_id,
      "pca_features": pca_features,
  }

  try:
    response = requests.post(api_url, json=payload, timeout=5)

    if response.status_code == 200:
      res_data = response.json()

      col1, col2, col3, col4 = st.columns(4)
      col1.metric("Fraud Decision", "FRAUD" if res_data["is_fraud"] else "LEGIT")
      col2.metric("Fraud Probability", f"{res_data['fraud_probability']:.2%}")
      col3.metric("Decision Threshold", f"{res_data['decision_threshold']:.2f}")
      col4.metric("Inference Latency", f"{res_data['latency_ms']} ms")

      if res_data["is_fraud"]:
        st.error("🚨 Warning: High-risk fraud pattern detected!")
      else:
        st.success("✅ Transaction verified: Low risk.")

      
      st.subheader("📊 SHAP Feature Attribution (Why this decision?)")

     
      scaled_amount = scaler_amount.transform([[amount]])[0][0]
      scaled_time = scaler_time.transform([[time_sec]])[0][0]

      card_pr = graph_lookup["pagerank"].get(card_id, 0.0)
      merchant_pr = graph_lookup["pagerank"].get(merchant_id, 0.0)
      card_dc = graph_lookup["degree_centrality"].get(card_id, 0.0)
      merchant_dc = graph_lookup["degree_centrality"].get(merchant_id, 0.0)

      input_row = {}
      for i in range(1, 29):
        input_row[f"V{i}"] = pca_features[f"V{i}"]
      input_row["scaled_amount"] = scaled_amount
      input_row["scaled_time"] = scaled_time
      input_row["card_pagerank"] = card_pr
      input_row["merchant_pagerank"] = merchant_pr
      input_row["card_degree_centrality"] = card_dc
      input_row["merchant_degree_centrality"] = merchant_dc

      df_input = pd.DataFrame([input_row])[feature_names]

      shap_values = explainer(df_input)

     
      st.markdown(
          "The plot below highlights features pushing the prediction toward"
          " **Fraud** (red) or **Legitimate** (blue):"
      )

      import matplotlib.pyplot as plt

      fig, ax = plt.subplots(figsize=(10, 5))
      shap.plots.waterfall(shap_values[0], max_display=10, show=False)
      st.pyplot(fig)

    else:
      st.error(f"API Error ({response.status_code}): {response.text}")

  except requests.exceptions.ConnectionError:
    st.error(
        f"Could not connect to FastAPI at `{api_url}`. Make sure `uvicorn` is"
        " running in another terminal!"
    )