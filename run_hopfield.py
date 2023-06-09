import random

import numpy as np
from matplotlib import pyplot as plt

import plots
import utils
from src.hopfield.hopfield import HopfieldNetwork
from src.hopfield.letters import *

# Define the alphabet matrices

espureo = [-1, 1, 1, 1, -1,
           1, -1, -1, -1, 1,
           -1, 1, -1, 1, -1,
           1, -1, -1, -1, 1,
           -1, 1, -1, 1, -1]

# Combine the matrices into a single vector
letters = [J, A, Z, B]
orthogonal = [P, M, L, I]

def test():
    TEST_COUNT = 100
    n = len(letters[0])
    h_not_ortho = HopfieldNetwork(n)
    h_not_ortho.train(letters)
    h_ortho = HopfieldNetwork(n)
    h_ortho.train(orthogonal)

    ortho_mean = []
    ortho_std = []
    not_ortho_mean = []
    not_ortho_std = []
    for i in range(0, n):
        not_ortho = []
        ortho = []
        for _ in range(TEST_COUNT):
            letter = random.choice(letters)
            noisy_letter = utils.add_small_noise(letter, i)
            recalled_letter, terminated = h_not_ortho.recall(noisy_letter, 100)
            recalled_letter = list(recalled_letter[-1].flatten())
            not_ortho.append(1 if terminated and letter == recalled_letter else 0)

            letter = random.choice(orthogonal)
            noisy_letter = utils.add_small_noise(letter, i)
            recalled_letter, terminated = h_ortho.recall(noisy_letter, 100)
            recalled_letter = list(recalled_letter[-1].flatten())
            ortho.append(1 if terminated and letter == recalled_letter else 0)
        ortho_mean.append(np.mean(ortho))
        ortho_std.append(np.std(ortho))
        not_ortho_mean.append(np.mean(not_ortho))
        not_ortho_std.append(np.std(not_ortho))
    print(ortho_mean)
    print(not_ortho_mean)
    fig, ax = plt.subplots()
    plt.title("Aciertos vs. #Celdas cambiadas (promedio de 100 experimentos)")
    plt.xlabel("Cantidad de celdas cambiadas")
    plt.ylabel("Rate de aciertos")
    ax.bar(np.arange(n), ortho_mean, alpha=0.75, label=r"$\Sigma |<,>| = 8$")
    ax.bar(np.arange(n), not_ortho_mean, alpha=0.75, label=r"$\Sigma |<,>| = 40$")
    plt.legend()
    plt.show()


def test2():
    TEST_COUNT = 100
    n = len(letters[0])
    h_4 = HopfieldNetwork(n)
    h_4.train(orthogonal)
    h_5 = HopfieldNetwork(n)
    extra = orthogonal + [A]
    h_5.train(extra)

    p, t = h_5.recall(A, 100)
    print("AAA" if t and list(p[-1].flatten()) == M else "NOOO")
    mean_4 = []
    mean_5 = []
    fig, ax = plt.subplots()
    for i in range(0, n):
        four = []
        five = []
        for _ in range(TEST_COUNT):
            letter = random.choice(orthogonal)
            noisy_letter = utils.add_small_noise(letter, i)
            recalled_letter, terminated = h_4.recall(noisy_letter, 100)
            recalled_letter = list(recalled_letter[-1].flatten())
            four.append(1 if terminated and letter == recalled_letter else 0)

            letter = random.choice(extra)
            noisy_letter = utils.add_small_noise(letter, i)
            recalled_letter, terminated = h_5.recall(noisy_letter, 100)
            recalled_letter = list(recalled_letter[-1].flatten())
            five.append(1 if terminated and letter == recalled_letter else 0)
        mean_4.append(np.mean(four))
        mean_5.append(np.mean(five))
    print(mean_4)
    print(mean_5)
    plt.title("Aciertos vs. #Celdas cambiadas (promedio de 1000 experimentos)")
    plt.xlabel("Cantidad de celdas cambiadas")
    plt.ylabel("Rate de aciertos")
    ax.bar(np.arange(n), mean_4, alpha=0.75, label=r"#I = 4")
    ax.bar(np.arange(n), mean_5, alpha=0.75, label=r"#I = 5")
    plt.legend()
    plt.show()


def main():
    test2()
    hopfield = HopfieldNetwork(len(letters[0]))
    hopfield.train(letters)

    letter_with_noise = utils.add_small_noise(L, 0)

    pattern_history, terminated = hopfield.recall(letter_with_noise, 10)
    energy_history = [hopfield.energy(pattern) for pattern in pattern_history]
    pattern_history = [pattern.reshape(5, 5) for pattern in pattern_history]
    for pattern in pattern_history:
        plots.pattern(pattern)


if __name__ == "__main__":
    main()
