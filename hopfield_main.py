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

    print(hopfield.recall(A, 100) == A)


if __name__ == "__main__":
    main()
