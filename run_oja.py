import plots
import utils
from src.oja.learning_rate_supplier import ConstantLearningRate
from src.oja.oja import OjaRule


def main():
    variables, countries, data = utils.read_data_from_csv("./data/europe.csv")
    standardized_data = utils.standarize_matrix_by_colum(data)

    w = OjaRule.initialize_weights(data.shape[1])
    OjaRule.train(data=standardized_data,
                  learning_rate_supplier=ConstantLearningRate(0.001),
                  weights=w,
                  epochs=1000)
    print(w)
    scores = [OjaRule.test(data, w) for data in standardized_data]
    sorted_scores, sorted_countries = zip(*sorted(zip(scores, countries)))
    plots.bargraph(sorted_scores, sorted_countries, title="Components with Oja Rule", x_label="PC1",
                   y_label="Countries")


if __name__ == '__main__':
    main()
