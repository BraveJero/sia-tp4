import pandas as pd
import numpy as np
from numpy import ndarray


def read_data_from_csv(filename: str, sep: str = ',', header: int = 0):
    df = pd.read_csv(filename, sep=sep, header=header)
    return df.columns.values.tolist(), df.values[:, 0].tolist(), df.values[:, 1:]
