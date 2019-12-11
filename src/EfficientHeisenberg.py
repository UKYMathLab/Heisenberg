import argparse
import itertools as it
from math import factorial

import numpy as np
from tqdm import tqdm
import pptk

# for 3D printing
# from stl import mesh

from HeisenbergCompute import HeisenbergVectors
from utils import point_cloud
import drivers


def choose_basis():

    choice = input('Which basis?\nChoices: standard (s), weird (w), real (r)\n')

    # starting matrices
    if choice == 's':
        chosen_basis = np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=int)
    elif choice == 'w':
        chosen_basis = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, -1],
                                 [-1, 0, 1],
                                 [0, 0, 0]], dtype=int)
    if choice == 'r':
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
        chosen_basis = np.unique(real_basis, axis=0)

    return chosen_basis


def compute(chosen_basis: np.array):

    pt_size = 0.25
    # point_cloud(basis, pt_size)
    vectors = HeisenbergVectors(basis, num_sums=num_sums)
    # drivers.ExamineData(vectors.basis_vectors, 'Basis Vectors')

    # find all possible combinations of basis vectors
    vectors.compute_permutations()
    vectors.get_unique_permutations()
    drivers.ExamineData(vectors.unique_permutations, 'unique permutations')

    point_cloud(vectors.unique_permutations, pt_size)
    # plot_points(vectors.unique_permutations)


########################################################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sums', type=int, default=3, help='Number of sums to compute.')
    args = parser.parse_args()

    num_sums = args.num_sums

    basis = choose_basis()

    compute(basis)
