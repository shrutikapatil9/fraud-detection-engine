import joblib
import networkx as nx
import numpy as np
import pandas as pd


def generate_synthetic_graph_data(
    df: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:

  np.random.seed(seed)
  n_samples = len(df)

 
  card_ids = [f"card_{i}" for i in np.random.randint(1000, 6000, size=n_samples)]
  merchant_ids = [
      f"merchant_{i}" for i in np.random.randint(100, 1100, size=n_samples)
  ]

  df_copy = df.copy()
  df_copy["card_id"] = card_ids
  df_copy["merchant_id"] = merchant_ids

  return df_copy


def extract_graph_features(
    df: pd.DataFrame, is_training: bool = True
) -> pd.DataFrame:

  print("Constructing transaction network graph...")


  G = nx.Graph()

 
  edges = df.groupby(["card_id", "merchant_id"]).size().reset_index(name="weight")
  for _, row in edges.iterrows():
    G.add_edge(row["card_id"], row["merchant_id"], weight=row["weight"])

  print(
      f"Graph created with {G.number_of_nodes()} nodes and"
      f" {G.number_of_edges()} edges."
  )

  print("Computing PageRank & Degree Centrality...")

  pagerank = nx.pagerank(G, max_iter=200)


  degree_centrality = nx.degree_centrality(G)


  df_features = df.copy()
  df_features["card_pagerank"] = df_features["card_id"].map(pagerank).fillna(0)
  df_features["merchant_pagerank"] = (
      df_features["merchant_id"].map(pagerank).fillna(0)
  )

  df_features["card_degree_centrality"] = (
      df_features["card_id"].map(degree_centrality).fillna(0)
  )
  df_features["merchant_degree_centrality"] = (
      df_features["merchant_id"].map(degree_centrality).fillna(0)
  )

  if is_training:
  
    graph_lookup = {
        "pagerank": pagerank,
        "degree_centrality": degree_centrality,
    }
    joblib.dump(graph_lookup, "models/graph_lookup.pkl")
    print("Graph metrics saved to models/graph_lookup.pkl")

 
  df_features = df_features.drop(columns=["card_id", "merchant_id"])

  return df_features


if __name__ == "__main__":
  from preprocess import load_and_preprocess_data


  X_train, X_test, y_train, y_test = load_and_preprocess_data()

  
  X_train_graph = generate_synthetic_graph_data(X_train)
  X_train_enhanced = extract_graph_features(X_train_graph, is_training=True)

  print(
      f"Enhanced training dataset shape: {X_train_enhanced.shape} (Added 4"
      " Graph Features)"
  )