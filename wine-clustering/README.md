# Wine Clustering

Group 178 wines by chemistry with no labels, then check the groups against the real grape cultivars.

- **Data:** UCI Wine dataset, 13 chemical measurements. A public benchmark bundled with scikit-learn.
- **Method:** standardise the features, run KMeans for k = 2 to 8, pick k by silhouette score, project to 2D with PCA for plotting. Cultivar labels are used only after fitting.

## Results
- k = 3 has the highest silhouette score (0.285).
- Adjusted Rand index against the true cultivars: **0.90** (1.0 is a perfect match).
- Cluster sizes: 65, 51, 62.
- Features separating the clusters most: flavanoids, OD280/OD315 ratio, proline, total phenols, colour intensity.
- The first two principal components keep 55.4% of the variation.

## Limits
A silhouette of 0.29 means the groups touch rather than sit far apart. The match to the cultivars is strong, not perfect.

## Run
```bash
pip install scikit-learn numpy
python cluster.py
```
