from src.hopfield.hopfield import HopfieldNetwork


# Define the alphabet matrices
J = [1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1]

A = [1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1]

Z = [1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, 1, 1,
     -1, -1, 1, -1, 1,
     1, 1, 1, 1, 1]

B = [1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, -1]

# Combine the matrices into a single vector
letters = [J, A, Z]


def main():
    hopfield = HopfieldNetwork(len(letters[0]))
    hopfield.train(letters)

    pattern_history, terminated = hopfield.recall(B, 100)
    energy_history = [hopfield.energy(pattern) for pattern in pattern_history]
    pattern_history = [pattern.reshape(5, 5) for pattern in pattern_history]
    print(pattern_history[-1])


if __name__ == "__main__":
    main()
