import utils
import plots


def main():
    variables, countries, data = utils.read_data_from_csv("data/europe.csv")
    plots.boxplot(data, labels=variables[1:], title="Not standarized variables", x_label="Variables")
    standarized = utils.standarize_matrix_by_colum(data)
    plots.boxplot(standarized, labels=variables[1:], title="Standarized variables", x_label="Variables")


if __name__ == '__main__':
    main()
