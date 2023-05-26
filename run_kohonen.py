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

    plots.heatmap(hit_matrix, "hits", hit_matrix)


if __name__ == '__main__':
    main()
