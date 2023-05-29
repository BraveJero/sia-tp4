import itertools

import numpy as np

A = [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1]

B = [1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, -1]

C = [-1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     -1, 1, 1, 1, -1]

D = [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1]

E = [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1]

F = [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1]

G = [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, -1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1]

H = [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1]

I = [-1, -1, 1, -1, - 1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1]

J = [1, 1, 1, 1, 1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     1, -1, 1, -1, -1,
     1, 1, 1, -1, -1]

K = [1, -1, -1, 1, -1,
     1, -1, 1, -1, -1,
     1, 1, -1, -1, -1,
     1, -1, 1, -1, -1,
     1, -1, -1, 1, -1]

L = [1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1]

M = [1, -1, -1, -1, 1,
     1, 1, -1, 1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1]

N = [1, -1, -1, -1, 1,
     1, 1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1]

O = [1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1]

P = [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1]

Q = [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, 1, -1, -1,
     1, 1, -1, 1, -1,
     1, 1, 1, 1, 1]

R = [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, 1, -1, -1,
     1, -1, -1, 1, -1]

S = [-1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     -1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1,
     1, 1, 1, 1, -1]

T = [1, 1, 1, 1, 1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1]

U = [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1]

V = [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1]

W = [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, 1, -1, 1, 1,
     1, -1, -1, -1, 1]

X = [1, -1, -1, -1, 1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, 1, -1, 1, -1,
     1, -1, -1, -1, 1]

Y = [1, -1, -1, -1, 1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1]

Z = [1, 1, 1, 1, 1,
     -1, -1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, 1, -1, -1, -1,
     1, 1, 1, 1, 1]

l = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": H, "I": I, "J": J, "K": K, "L": L, "M": M, "N": N,
     "O": O, "P": P, "Q": Q, "R": R, "S": S, "T": T, "U": U, "V": V, "W": W, "X": X, "Y": Y, "Z": Z}


def get_letter_combinations_orthogonality():
    results = []
    for letters in itertools.permutations(l.keys(), 4):
        if letters not in results:
            results.append(letters)

            sum = 0

            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    dot = np.dot(l[letters[i]], l[letters[j]])
                    sum += np.abs(dot)
            if sum < 12:
                print(letters, sum)
