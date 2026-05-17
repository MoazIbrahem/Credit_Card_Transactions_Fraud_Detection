# CreditSeg AI — Customer Segmentation Dashboard

A professional Plotly Dash GUI for credit card customer segmentation
using K-Means, AE+K-Means, DBSCAN, and GMM clustering algorithms.

---

## Project Structure

```
Credit_Card_Transactions_Fraud_Detection/
│
├── app.py                        ← Entry point
├── requirements.txt
├── README.md
│
├── assets/
│   └── style.css                 ← ALL styling (no inline styles in Python)
│
├── pages/
│   ├── prediction.py             ← Prediction page layout + callbacks
│   └── comparison.py             ← Model comparison page layout + callbacks
│
├── components/
│   └── cards.py                  ← Reusable UI components
│
├── utils/
│   └── load_models.py            ← Model loading, DBSCAN fallback, AE wrapper
│
├── saved_models/
│   ├── kmeans_model.pkl
│   ├── ae_kmeans_model.pkl
│   ├── autoencoder.pt
│   ├── dbscan_model.pkl
│   ├── dbscan_nn.pkl             ← NearestNeighbors fallback for DBSCAN
│   ├── dbscan_labels.pkl
│   ├── gmm_model.pkl
│   ├── som_model.pkl             ← trained MiniSom object
│   ├── som_bmu_labels.pkl        ← 2D grid mapping BMU → cluster ID
│   └── som_grid.pkl              ← (grid_x, grid_y) tuple e.g. (20, 20)
│
├── preprocessing/
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── cluster_names.pkl
│
└── metrics/
    └── clustering_metrics.json   ← Pre-computed metrics for comparison page
```

---

## Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Save models from notebook
Add the saving cell (provided separately) to the end of your notebook and run it.
All `.pkl`, `.pt`, and `.json` files will be placed in the correct folders.

**Extra cells to add for SOM** (run after training the SOM in the notebook):

```python
# After training your SOM and running KMeans on BMU positions:

# 1. Save the SOM object
joblib.dump(som, 'saved_models/som_model.pkl')

# 2. KMeans on BMU positions to get discrete cluster labels per neuron
from sklearn.cluster import KMeans as _KM
bmu_positions = np.array([som.winner(x) for x in X_scaled])
n_som_clusters = 5  # adjust to your elbow/silhouette result
km_bmu = _KM(n_clusters=n_som_clusters, random_state=42, n_init=10)
km_bmu.fit(bmu_positions)

# 3. Build the 2D grid of cluster labels (shape = grid_x × grid_y)
grid_x, grid_y = 20, 20
bmu_grid_labels = np.full((grid_x, grid_y), -1)
for gx in range(grid_x):
    for gy in range(grid_y):
        bmu_grid_labels[gx, gy] = km_bmu.predict([[gx, gy]])[0]

joblib.dump(bmu_grid_labels, 'saved_models/som_bmu_labels.pkl')
joblib.dump((grid_x, grid_y), 'saved_models/som_grid.pkl')

# 4. SOM cluster naming
som_cluster_labels_per_point = km_bmu.predict(bmu_positions)
profile_som_orig = df_clean.drop(columns=['CUST_ID'], errors='ignore').copy()
profile_som_orig['Cluster'] = som_cluster_labels_per_point
profile_som_means = profile_som_orig.groupby('Cluster').mean()
som_cluster_names = auto_name_clusters(profile_som_means)
cluster_names['som'] = som_cluster_names
joblib.dump(cluster_names, 'preprocessing/cluster_names.pkl')  # re-save with SOM included

# 5. Add SOM to metrics JSON
som_sil = silhouette_score(X_scaled, som_cluster_labels_per_point)
som_dbi = davies_bouldin_score(X_scaled, som_cluster_labels_per_point)
som_ch  = calinski_harabasz_score(X_scaled, som_cluster_labels_per_point)
metrics_data['SOM'] = {
    'silhouette': round(som_sil, 4),
    'davies_bouldin': round(som_dbi, 4),
    'calinski_harabasz': round(som_ch, 2),
    'n_clusters': n_som_clusters,
    'noise_pct': 0.0
}
with open('metrics/clustering_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)
print("✅ SOM objects saved.")
```

### 3. Run the app
```bash
python app.py
```
Then open `http://127.0.0.1:8050` in your browser.

### 4. Production deployment (gunicorn)
```bash
gunicorn app:server -b 0.0.0.0:8050 --workers 2
```

---

## Pages

### Prediction Page
- Enter 17 behavioral features for a credit card customer
- Select one or more clustering models
- View segment assignment (cluster ID + business name) per model
- Color-coded confidence bar per model

### Comparison Page
- Toggle which models to include
- Best-model banner (composite rank across all metrics)
- KPI cards per model
- Bar charts: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Cluster properties table: number of clusters, noise %
- Normalised radar chart for visual overview

---

## Model Notes

| Algorithm    | Prediction Strategy               | Notes                                          |
|---|---|---|
| K-Means      | Direct `.predict()`               | Standard centroid assignment                   |
| AE + K-Means | Encode → K-Means `.predict()`     | PyTorch encoder + KMeans chained               |
| DBSCAN       | NearestNeighbors fallback         | DBSCAN cannot predict new points natively      |
| GMM          | Direct `.predict()`               | Soft probabilities available                   |
| SOM          | BMU lookup → cluster label        | Finds Best Matching Unit, maps to cluster ID   |

---

## Feature Input Order (must match training)

```
BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE
```
