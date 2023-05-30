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


def main():
    hopfield = HopfieldNetwork(len(letters[0]))
    hopfield.train(letters)

    letter_with_noise = utils.add_small_noise(Z, 3)

    pattern_history, terminated = hopfield.recall(letter_with_noise, 10)
    energy_history = [hopfield.energy(pattern) for pattern in pattern_history]
    pattern_history = [pattern.reshape(5, 5) for pattern in pattern_history]
    for pattern in pattern_history:
        plots.pattern(pattern)


if __name__ == "__main__":
    main()
