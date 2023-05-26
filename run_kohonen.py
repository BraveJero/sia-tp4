import numpy as np

import plots
import utils
from src.kohonen.kohonen import KohonenNetwork


def print_top_10_distances():
    variables, countries, data = utils.read_data_from_csv("./data/europe.csv")
    standardized = utils.standarize_matrix_by_colum(data)

    # Calculate the pairwise Euclidean distances between all countries
    n = len(standardized)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(standardized[i] - standardized[j])
            dists.append((dist, countries[i], countries[j]))

    # Sort the distances in ascending order
    dists.sort()

    # Print out the top 10 distances and the corresponding country pairs
    print("Top 10 closest country pairs based on standardized data:")
    for i, (dist, country1, country2) in enumerate(dists[:10]):
        print(f"{i + 1}. {country1} - {country2}: {dist}")


def main():
    variables, countries, data = utils.read_data_from_csv("./data/europe.csv")
    standardized = utils.standarize_matrix_by_colum(data)

    size = 4
    kohonen = KohonenNetwork(size=size, weights=standardized)

    kohonen.train(standardized)

    hit_matrix = np.zeros((size, size))
    names_matrix = [["" for j in range(size)] for i in range(size)]

    for idx, element in enumerate(standardized):
        i, j = kohonen.get_closest_weight_to_element_index(element)
        hit_matrix[i, j] += 1
        names_matrix[i][j] += countries[idx] + "\n"

    plots.heatmap(hit_matrix, "hits.png", "hits", names_matrix)

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
    print_top_10_distances()
