import csv
import json
import sys

import numpy as np
from numpy import ndarray


def get_settings():
    if len(sys.argv) < 2:
        print("Config file argument not found")
        exit(1)

    path = sys.argv[1]
    with open(path, "r") as f:
        settings = json.load(f)
    if settings is None:
        raise ValueError("Unable to open settings")
    return settings


def read_data_from_csv(filename: str, sep: str = ',', header: int = 0):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        rows = list(reader)

    headers = rows[header]
    labels = [row[0] for row in rows[1:]]
    values = np.array([row[1:] for row in rows[1:]], dtype=float)

    return headers, labels, values


def standarize_matrix_by_colum(m: ndarray) -> ndarray:
    return (m - np.mean(m, axis=0)) / np.std(m.astype(float), axis=0)

# def visualize_clusters(clusters, cluster_centroids, dim_a, dim_b):
#     som_shape = (dim_a, dim_b)  # Adjust the shape of the SOM grid if necessary
#     print(clusters)
#     som = MiniSom(som_shape[0], som_shape[1], 7)  # 7 Number of columns in europe.csv
#     print(cluster_centroids)
#     som.pca_weights_init(cluster_centroids)
#
#     plt.figure(figsize=(dim_a, dim_b))
#     plt.pcolor(som.distance_map().T, cmap='bone_r')
#
#     for i, cluster in enumerate(clusters):
#         cluster_coords = [som.winner(c)[0] for c in cluster]
#         x = cluster_coords[0]
#         y = cluster_coords[1]
#         plt.plot(x, y, 'C0', marker='o', markersize=8, alpha=0.6)
#
#     plt.colorbar()
#     plt.show()
