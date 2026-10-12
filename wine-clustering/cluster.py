"""Group 178 wines by chemistry without using the cultivar labels.

Real public benchmark data (UCI Wine, bundled with scikit-learn). KMeans is
fitted on standardised chemistry only; the true cultivar labels are used
afterwards, purely to check how well the clusters recovered them.
"""

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
data = load_wine()
X, truth, names = data.data, data.target, list(data.feature_names)
Z = StandardScaler().fit_transform(X)

scan = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
    scan.append(
        {
            "k": k,
            "inertia": round(float(km.inertia_), 1),
            "silhouette": round(float(silhouette_score(Z, km.labels_)), 3),
        }
    )

best_k = max(scan, key=lambda s: s["silhouette"])["k"]
km = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(Z)
labels = km.labels_

pca = PCA(n_components=2, random_state=SEED)
P = pca.fit_transform(Z)

# Which features separate the clusters most (spread of cluster centres, in std units)
centres = km.cluster_centers_
spread = centres.max(axis=0) - centres.min(axis=0)
top = sorted(zip(names, spread), key=lambda t: -t[1])[:5]

alcohol_idx = names.index("alcohol")
points = [
    [round(float(P[i, 0]), 3), round(float(P[i, 1]), 3), int(labels[i]), int(truth[i]), round(float(X[i, alcohol_idx]), 2)]
    for i in range(len(X))
]

out = {
    "n_samples": int(len(X)),
    "n_features": int(X.shape[1]),
    "k_scan": scan,
    "best_k_by_silhouette": int(best_k),
    "chosen_k": 3,
    "silhouette_k3": round(float(silhouette_score(Z, labels)), 3),
    "adjusted_rand_vs_cultivar": round(float(adjusted_rand_score(truth, labels)), 3),
    "pca_variance_explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
    "top_separating_features": [[n, round(float(v), 2)] for n, v in top],
    "cluster_sizes": [int((labels == c).sum()) for c in range(3)],
    "points": points,  # x, y, cluster, true cultivar, alcohol
}
json.dump(out, open("results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "points"}, indent=2))
