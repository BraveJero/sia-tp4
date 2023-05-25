import numpy as np
from src.kohonen import KohonenNet
from utils import visualize_clusters


def main():
    kohonen_net = KohonenNet(4, 4, 10000)
    cluster, cluster_centroids = kohonen_net.train_min_som("./data/europe.csv")

    visualize_clusters(cluster, cluster_centroids, 4, 4)


if __name__ == '__main__':
    main()
