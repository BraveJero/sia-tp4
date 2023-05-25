import numpy as np

import utils
from minisom import MiniSom


class KohonenNet:
    def __init__(self, map_dim_a: int, map_dim_b: int, iterations: int):
        self._map_dim_a = map_dim_a
        self._map_dim_b = map_dim_b
        self._iter = iterations

    def train_min_som(self, data_csv: str):
        variables, countries, data = utils.read_data_from_csv(data_csv)
        standarized = utils.standarize_matrix_by_colum(data)

        som_shape = (self._map_dim_a, self._map_dim_b)
        som = MiniSom(som_shape[0], som_shape[1], standarized.shape[1])
        som.random_weights_init(standarized)
        som.train_random(standarized, self._iter)

        clusters = som.labels_map(standarized, countries)
        cluster_centroids = np.array([np.mean(standarized[np.where(clusters == label)], axis=0) for label in np.unique(clusters)])

        return clusters, cluster_centroids



