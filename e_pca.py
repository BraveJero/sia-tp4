import utils
import plots
from sklearn.decomposition import PCA


def main():
    variables, countries, data = utils.read_data_from_csv("data/europe.csv")
    plots.boxplot(data, labels=variables[1:], title="Not standarized variables", x_label="Variables")
    standarized = utils.standarize_matrix_by_colum(data)
    plots.boxplot(standarized, labels=variables[1:], title="Standarized variables", x_label="Variables")
    pca = PCA(n_components=2)
    pca.fit(standarized)
    print(f"PCA components: {pca.components_}")
    pcs = pca.transform(standarized)
    print(f"pc1 y pc2: {pcs}")
    plots.bargraph(pcs[:, 0], countries, title="", x_label="Countries", y_label="PC1")


if __name__ == '__main__':
    main()
