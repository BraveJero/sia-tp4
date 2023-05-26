import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def read_data_from_csv(filename: str, sep: str = ',', header: int = 0):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        rows = list(reader)

    headers = rows[header]
    labels = [row[0] for row in rows[1:]]
    values = np.array([row[1:] for row in rows[1:]], dtype=float)

    return headers, labels, values


def standarize_matrix_by_colum(m: ndarray) -> ndarray:
    return (m - np.mean(m, axis=0)) / np.std(m.astype(float), axis=0)
