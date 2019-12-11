from math import factorial
import itertools as it

import numpy as np
from tqdm import tqdm

import drivers


class HeisenbergVectors():
    def __init__(self, basis_vectors: np.array, num_sums: int):
        self.basis_vectors = basis_vectors
        self.num_sums = num_sums

        self.all_permutations = []
        self.unique_permutations = np.zeros((1, 3))

    @property
    def num_permutations(self):
        return len(self.all_permutations)

    @property
    def num_unique_permutations(self):
        return self.unique_permutations.shape[0]

    def compute_permutations(self):
        r"""Computes all permutations of the rows of an array.

        Returns a list of tuples of np.array representing all the permutations.
        """

        # compute all permutations of rows
        vectors = np.vsplit(self.basis_vectors, self.basis_vectors.shape[0])
        permutations = tqdm(it.product(vectors, repeat=self.num_sums),
                                total=len(vectors)**self.num_sums,
                                desc='Computing permutations')
        self.all_permutations = [np.vstack(tup) for tup in permutations]

        # check computations went correctly
        # drivers.ExamineData(self.all_permutations[0], 'sample permutation')
        drivers.CheckValidLength(len(vectors), self.basis_vectors.shape[0], mode='equal')
        drivers.CheckValidLength(len(permutations), len(vectors)**self.num_sums, mode='equal')

    def get_unique_permutations(self):
        r"""Finds all unique elements of an array."""

        # for permutation in self.all_permutations:
        #     ExamineData(permutation, 'permutation')
        summed = list(tqdm(map(_permutation_sum, self.all_permutations),
                           total=self.num_permutations,
                           desc='Summing permutations'))

        self.unique_permutations = np.unique(np.vstack(summed), axis=0)

        # check computations went correctly
        drivers.CheckValidLength(self.num_unique_permutations, self.num_permutations,
                                 mode='less than', strict=False)


def _heisenberg_sum(u: np.array, v: np.array) -> np.array:
    r"""Computes the Heisenberg sum of two vectors.

    The Heisenberg sum of two vectors is defined as:
        u @ v = (x, y, z) @ (x', y', z') := (x + x', y + y', z + z' + x * y')
    """

    w = u + v
    w[2] += u[0] * v[1]

    return w


def _permutation_sum(permutation: np.array) -> np.array:

    # ExamineData(permutation, 'Permutation')
    max_iter = permutation.shape[0] - 1
    # check computations went correctlyi
    drivers.CheckValidLength(max_iter, 0, mode='greater than')

    for i  in range(max_iter):
        # make sure in valid range (since operating on pairs)
        intermediate_result = _heisenberg_sum(permutation[i], permutation[i+1])
        permutation[i+1] = intermediate_result
    # drivers.ExamineData(intermediate_result, 'intermediate').pause()

    return intermediate_result
