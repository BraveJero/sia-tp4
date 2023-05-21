from utils import read_data_from_csv


def main():
    variables, countries, data = read_data_from_csv("data/europe.csv")
    print(variables)
    print(countries)
    print(type(data))
    print(data)


if __name__ == '__main__':
    main()
