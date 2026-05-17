import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Autoencoder Architecture (must match the one trained in the notebook)
class Autoencoder(nn.Module):
    def __init__(self, input_dim=17, encoding_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# Prediction Wrappers
class DBSCANPredictor:
    """
    DBSCAN cannot predict new points natively.
    We use a NearestNeighbors fallback: assign new input to the cluster
    of its closest training point.
    """
    def __init__(self, nn_model, training_labels):
        self.nn_model = nn_model
        self.training_labels = np.array(training_labels)

    def predict(self, X):
        distances, indices = self.nn_model.kneighbors(X)
        return self.training_labels[indices.flatten()]

    def predict_proba(self, X):
        distances, _ = self.nn_model.kneighbors(X)
        confidence = 1 / (1 + distances.flatten())
        return np.column_stack([1 - confidence, confidence])


class AEKMeansPredictor:
    """
    Chains PyTorch encoder → KMeans predict.
    """
    def __init__(self, autoencoder, kmeans):
        self.autoencoder = autoencoder
        self.kmeans = kmeans

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.autoencoder.encode(X_tensor).numpy()
        return self.kmeans.predict(encoded)

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.autoencoder.encode(X_tensor).numpy()
        distances = np.linalg.norm(
            encoded[:, np.newaxis] - self.kmeans.cluster_centers_, axis=2
        )
        inv_dist = 1 / (distances + 1e-9)
        proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        return proba


class SOMPredictor:
    """
    SOM wrapper: finds the Best Matching Unit (BMU) for a new input,
    then returns the cluster label assigned to that BMU during training.
    """
    def __init__(self, som, bmu_cluster_labels, grid_x, grid_y):
        self.som = som
        self.bmu_cluster_labels = np.array(bmu_cluster_labels).reshape(grid_x, grid_y)
        self.grid_x = grid_x
        self.grid_y = grid_y

    def predict(self, X):
        labels = []
        for row in X:
            bmu_x, bmu_y = self.som.winner(row)
            labels.append(int(self.bmu_cluster_labels[bmu_x, bmu_y]))
        return np.array(labels)

    def predict_proba(self, X):
        confidences = []
        for row in X:
            bmu_x, bmu_y = self.som.winner(row)
            weight = self.som.get_weights()[bmu_x, bmu_y]
            qe = np.linalg.norm(row - weight)
            conf = 1 / (1 + qe)
            confidences.append(conf)
        conf_arr = np.array(confidences)
        return np.column_stack([1 - conf_arr, conf_arr])



# Helper: safely load a file, return None if missing

def _safe_load(path, loader=joblib.load, **kwargs):
    if not os.path.exists(path):
        print(f"⚠️  Missing file (skipping): {path}")
        return None
    try:
        if loader == "json":
            with open(path, "r") as f:
                return json.load(f)
        return loader(path, **kwargs)
    except Exception as e:
        print(f"⚠️  Failed to load {path}: {e}")
        return None


# Main Load Function
def load_all():
    """
    Returns: models_dict, scaler, feature_columns, cluster_names, metrics

    Gracefully skips any model whose required files are missing,
    so the GUI always starts even on a partial save directory.
    """
    paths = {
        "scaler":          os.path.join(BASE_DIR, "preprocessing", "scaler.pkl"),
        "features":        os.path.join(BASE_DIR, "preprocessing", "feature_columns.pkl"),
        "cluster_names":   os.path.join(BASE_DIR, "preprocessing", "cluster_names.pkl"),
        "kmeans":          os.path.join(BASE_DIR, "saved_models",  "kmeans_model.pkl"),
        "ae_kmeans":       os.path.join(BASE_DIR, "saved_models",  "ae_kmeans_model.pkl"),
        "autoencoder":     os.path.join(BASE_DIR, "saved_models",  "autoencoder.pt"),
        "dbscan_nn":       os.path.join(BASE_DIR, "saved_models",  "dbscan_nn.pkl"),
        "dbscan_labels":   os.path.join(BASE_DIR, "saved_models",  "dbscan_labels.pkl"),
        "gmm":             os.path.join(BASE_DIR, "saved_models",  "gmm_model.pkl"),
        "som":             os.path.join(BASE_DIR, "saved_models",  "som_model.pkl"),
        "som_bmu_labels":  os.path.join(BASE_DIR, "saved_models",  "som_bmu_labels.pkl"),
        "som_grid":        os.path.join(BASE_DIR, "saved_models",  "som_grid.pkl"),
        "metrics":         os.path.join(BASE_DIR, "metrics",       "clustering_metrics.json"),
    }

    #  Required preprocessing (raise if truly missing) 
    scaler        = joblib.load(paths["scaler"])
    features      = joblib.load(paths["features"])
    cluster_names = joblib.load(paths["cluster_names"])

    models = {}

    #  Standard K-Means 
    kmeans_model = _safe_load(paths["kmeans"])
    if kmeans_model is not None:
        models["K-Means"] = kmeans_model
    else:
        print("⚠️  K-Means model not loaded — skipping.")

    #  AE + K-Means 
    ae_kmeans_model = _safe_load(paths["ae_kmeans"])
    if ae_kmeans_model is not None and os.path.exists(paths["autoencoder"]):
        try:
            input_dim   = len(features)
            autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=8)
            autoencoder.load_state_dict(
                torch.load(paths["autoencoder"], map_location="cpu")
            )
            autoencoder.eval()
            models["AE + K-Means"] = AEKMeansPredictor(autoencoder, ae_kmeans_model)
        except Exception as e:
            print(f"⚠️  AE + K-Means failed to load: {e}")
    else:
        print("⚠️  AE + K-Means model not loaded — skipping.")

    #  DBSCAN 
    # dbscan_nn.pkl and dbscan_labels.pkl are generated by the notebook
    # export cell (see instructions below if they are missing).
    dbscan_nn_obj    = _safe_load(paths["dbscan_nn"])
    dbscan_label_arr = _safe_load(paths["dbscan_labels"])

    if dbscan_nn_obj is not None and dbscan_label_arr is not None:
        models["DBSCAN"] = DBSCANPredictor(dbscan_nn_obj, dbscan_label_arr)
    else:
        #  Fallback: rebuild NN from the DBSCAN model's training data 
        # This requires the main DBSCAN model and the scaled training array.
        # If those are also missing we simply skip DBSCAN.
        dbscan_model = _safe_load(paths["dbscan"])  # the raw DBSCAN object
        if dbscan_model is not None and hasattr(dbscan_model, "labels_"):
            try:
                print("ℹ️  Rebuilding DBSCAN NN fallback from saved DBSCAN object…")
                # We need the original scaled data — attempt to load from CSV
                csv_path = os.path.join(BASE_DIR, "customer_segments_final.csv")
                if os.path.exists(csv_path):
                    import pandas as pd
                    df_csv = pd.read_csv(csv_path)
                    from sklearn.preprocessing import StandardScaler
                    feat_cols = [c for c in features if c in df_csv.columns]
                    X_train   = scaler.transform(df_csv[feat_cols].fillna(0))
                    nn_model  = NearestNeighbors(n_neighbors=1).fit(X_train)
                    models["DBSCAN"] = DBSCANPredictor(nn_model, dbscan_model.labels_)
                    print("✅  DBSCAN rebuilt from CSV.")
                else:
                    print("⚠️  DBSCAN skipped — no NN file and no training CSV found.")
            except Exception as e:
                print(f"⚠️  DBSCAN fallback rebuild failed: {e}")
        else:
            print("⚠️  DBSCAN skipped — required files not found.")

    #  GMM 
    gmm_model = _safe_load(paths["gmm"])
    if gmm_model is not None:
        models["GMM"] = gmm_model
    else:
        print("⚠️  GMM model not loaded — skipping.")

    #  SOM 
    som_model      = _safe_load(paths["som"])
    som_bmu_labels = _safe_load(paths["som_bmu_labels"])
    som_grid       = _safe_load(paths["som_grid"])

    if som_model is not None and som_bmu_labels is not None and som_grid is not None:
        grid_x, grid_y = som_grid
        models["SOM"] = SOMPredictor(som_model, som_bmu_labels, grid_x, grid_y)
    else:
        print("⚠️  SOM model not loaded — skipping.")

    #  Metrics 
    metrics = _safe_load(paths["metrics"], loader="json") or {}

    print(f"\n✅ Models loaded: {list(models.keys())}")
    return models, scaler, features, cluster_names, metrics