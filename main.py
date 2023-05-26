import utils
from src.kohonen.kohonen import KohonenNetwork


def main():
    variables, countries, data = utils.read_data_from_csv("./data/europe.csv")
    standardized = utils.standarize_matrix_by_colum(data)

    kohonen = KohonenNetwork(size=4, weights=standardized)

    kohonen.train(standardized)

    # visualize_clusters(cluster, cluster_centroids, 4, 4)


if __name__ == '__main__':
    main()
