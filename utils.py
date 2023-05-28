import csv
import json
import random
import sys

import numpy as np
from numpy import ndarray


def get_settings():
    if len(sys.argv) < 2:
        print("Config file argument not found")
        exit(1)

    path = sys.argv[1]
    with open(path, "r") as f:
        settings = json.load(f)
    if settings is None:
        raise ValueError("Unable to open settings")
    return settings


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


def add_noise(letters, gamma=0.5):
    letters_with_noise = [list(letter) for letter in letters]

    for letter in letters_with_noise:
        for i, cells in enumerate(letter):
            if random.random() <= gamma:
                letter[i] = letter[i] * -1

    return letters_with_noise
