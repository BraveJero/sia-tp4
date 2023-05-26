import numpy as np

import plots
import utils
from src.kohonen.kohonen import KohonenNetwork


def main():
    variables, countries, data = utils.read_data_from_csv("./data/europe.csv")
    standardized = utils.standarize_matrix_by_colum(data)

    size = 4
    kohonen = KohonenNetwork(size=size, weights=standardized)

    kohonen.train(standardized)

    hit_matrix = np.zeros((size, size))

    for element in standardized:
        i, j = kohonen.get_closest_weight_to_element_index(element)
        hit_matrix[i, j] += 1

    plots.heatmap(hit_matrix, "hits.png", "hits", hit_matrix)

    u_matrix = np.zeros((size, size))
    neighbor_indices = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(size):
        for j in range(size):
            distances = []
            for ni, nj in neighbor_indices:
                if 0 <= i + ni < size and 0 <= j + nj < size:  # Check if neighbor is within bounds
                    distances.append(np.linalg.norm(kohonen.matrix[i, j] - kohonen.matrix[i + ni, j + nj]))
            u_matrix[i, j] = np.mean(distances)

    plots.heatmap(u_matrix, "umatrix.png", "u-matrix")


if __name__ == '__main__':
    main()
