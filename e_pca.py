import utils
import plots
from sklearn.decomposition import PCA


def main():
    variables, countries, data = utils.read_data_from_csv("data/europe.csv")
    variables = variables[1:]
    plots.boxplot(data, labels=variables, title="Not standarized variables", x_label="Variables")
    standarized = utils.standarize_matrix_by_colum(data)
    plots.boxplot(standarized, labels=variables, title="Standarized variables", x_label="Variables")
    pca = PCA(n_components=2)
    pca.fit(standarized)
    print(f"PCA components: {pca.components_}")
    pcs = pca.transform(standarized)
    print(f"pc1 y pc2: {pcs}")
    sorted_pc1s, sorted_countries = zip(*sorted(zip(pcs[:, 0], countries)))
    plots.bargraph(sorted_pc1s, sorted_countries, title="Components with SKLearn", x_label="PC1", y_label="Countries")
    plots.biplot(pcs[:, 0], pcs[:, 1], countries, variables, pca.components_, title="Biplot")


if __name__ == '__main__':
    main()
