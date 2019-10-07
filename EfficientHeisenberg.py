# Created and maintained by Sean Grate

# Last updated: August 27th, 2019


########################################################################################################################

import argparse
import itertools as it
from math import factorial

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D , art3d

import pptk


# for 3D printing
# from stl import mesh


########################################################################################################################


class ExamineData():
    def __init__(self, data: np.array, data_name: str = ''):
        header = ''.join([data_name, '\n', '='*len(data_name)])
        print(f'{header}\n{data}')
        print(f'{data_name} shape: {data.shape}\n\n\n')
    def pause(self):
        input()

class CheckValidLength():
    def __init__(self, calculated, threshold, mode: str, strict: bool = True):
        if mode == 'less than':
            if strict:
                assert calculated < threshold, f'{calculated} >= {threshold} !'
            else:
                assert calculated <= threshold, f'{calculated} > {threshold} !'
        elif mode == 'greater than':
            if strict:
                assert calculated > threshold, f'{calculated} <= {threshold} !'
            else:
                assert calculated >= threshold, f'{calculated} < {threshold} !'
        elif mode == 'equal':
            assert calculated == threshold, f'{calculated} != {threshold} !'
        elif mode == 'not equal':
            assert calculated != threshold, f'{calculated} = {threshold} !'
        else:
            print('mode variable not valid! Validation skipped.')


########################################################################################################################


class HeisenbergVectors():
    def __init__(self, basis_vectors: np.array, num_sums: int):
        self.basis_vectors = basis_vectors
        self.num_sums = num_sums

        self.all_permutations = []
        self.num_permutations = 0
        self.unique_permutations = np.zeros((1, 3))
        self.num_unique_permutations = 0

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
        self.num_permutations = len(self.all_permutations)
        # check computations went correctly
        # ExamineData(self.all_permutations[0], 'sample permutation')
        CheckValidLength(len(vectors), self.basis_vectors.shape[0], mode='equal')
        CheckValidLength(len(permutations), len(vectors)**self.num_sums, mode='equal')

    def get_unique_permutations(self):
        r"""Finds all unique elements of an array."""

        # for permutation in self.all_permutations:
        #     ExamineData(permutation, 'permutation')
        summed = list(tqdm(map(permutation_sum, self.all_permutations),
                           total=self.num_permutations,
                           desc='Summing permutations'))

        self.unique_permutations = np.unique(np.vstack(summed), axis=0)
        self.num_unique_permutations = self.unique_permutations.shape[0]

        # check computations went correctly
        CheckValidLength(self.num_unique_permutations, self.num_permutations,
                         mode='less than', strict=False)


########################################################################################################################


def heisenberg_sum(u: np.array, v: np.array) -> np.array:
    r"""Computes the Heisenberg sum of two vectors.

    The Heisenberg sum of two vectors is defined as:
        u @ v = (x, y, z) @ (x', y', z') := (x + x', y + y', z + z' + x * y')
    """

    w = u + v
    w[2] += u[0] * v[1]

    return w


def permutation_sum(permutation: np.array) -> np.array:

    # ExamineData(permutation, 'Permutation')
    max_iter = permutation.shape[0] - 1
    # check computations went correctly
    CheckValidLength(max_iter, 0, mode='greater than')

    for i  in range(max_iter):
        # make sure in valid range (since operating on pairs)
        intermediate_result = heisenberg_sum(permutation[i], permutation[i+1])
        permutation[i+1] = intermediate_result
    # ExamineData(intermediate_result, 'intermediate').pause()

    return intermediate_result


def plot_points(points: np.array):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        # set a limit on the axes
        ax.set_ylim(points[:, 1].min(), points[:, 0].max())
        ax.set_xlim(points[:, 0].min(), points[:, 1].max())
        ax.set_zlim(points[:, 2].min(), points[:, 2].max())

        plt.show()

def point_cloud(points, size: float):

    v = pptk.viewer(points)
    v.attributes(points[:, 2])
    v.set(point_size=size, floor_level=0)
    v.wait()

def compute():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sums', type=int, default=3, help='Number of sums to compute.')
    args = parser.parse_args()

    num_sums = args.num_sums

    # starting matrices
    standard_basis = np.array([[0, 0, 0],
                               [1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype=int)
    weird_basis = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, -1],
                            [-1, 0, 1],
                            [0, 0, 0]], dtype=int)

    real_x = np.linspace(0, 1, num=10)
    real_y = np.linspace(0, 1, num=10)
    real_z = np.linspace(0, 1, num=10)

    real_basis = np.zeros((real_x.shape[0]*real_y.shape[0]*real_z.shape[0], 3))

    idx = 0
    for (x, y, z) in tqdm(it.product(np.nditer(real_x), np.nditer(real_y), np.nditer(real_z)),
                          desc='Generating real basis'):
        if x + y + z <= 1:
            real_basis[idx, 0] = x
            real_basis[idx, 1] = y
            real_basis[idx, 2] = z

        idx += 1
    real_basis = np.unique(real_basis, axis=0)

    # pt_size = 0.01
    pt_size = 0.25
    point_cloud(weird_basis, pt_size)
    vectors = HeisenbergVectors(weird_basis, num_sums=num_sums)
    ExamineData(vectors.basis_vectors, 'Basis Vectors')

    # find all possible combinations of basis vectors
    vectors.compute_permutations()
    vectors.get_unique_permutations()
    # ExamineData(vectors.unique_permutations, 'unique permutations')

    point_cloud(vectors.unique_permutations, pt_size)
    # plot_points(vectors.unique_permutations)


########################################################################################################################


if __name__ == '__main__':
    compute()
