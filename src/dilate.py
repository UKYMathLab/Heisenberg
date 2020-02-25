from configparser import ConfigParser
from pathlib import Path
import itertools as it
from math import factorial

import numpy as np
from tqdm import tqdm
import pptk

# for 3D printing
# from stl import mesh

from shapes import Point, Polytope, JoinedPolytope
import drivers


def choose_basis(config: ConfigParser):
    r"""Chooses a basis of points.

    :param config: ConfigParser
        Determines which basis to use. User specified in the "SETTINGS/basis"
        option in config.ini. Currently only "s", "w", "r" are supported.
        (1) "s" consists of the unit points of R^3.
        (2) "w" consists a an arbitrary number of integer points in R^3.
        (3) "r" consists of real points (not necessarily integral!).
    """

    choice = config["SETTINGS"]["basis"]
    # starting matrices
    if choice == "s":
        basis = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=int)
    elif choice == "w":
        basis = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1],
                          [0, 0, 0]], dtype=int)
    elif choice == "r":
        real_x = np.linspace(0, 1, num=10)
        real_y = np.linspace(0, 1, num=10)
        real_z = np.linspace(0, 1, num=10)

        real_basis = np.zeros((real_x.shape[0]*real_y.shape[0]*real_z.shape[0], 3))

        idx = 0
        for (x, y, z) in tqdm(it.product(np.nditer(real_x), np.nditer(real_y), np.nditer(real_z)),
                              desc="Generating real basis"):
            if x + y + z <= 1:
                real_basis[idx, 0] = x
                real_basis[idx, 1] = y
                real_basis[idx, 2] = z

            idx += 1
        basis = np.unique(real_basis, axis=0)

    else:
        raise NotImplementedError("Invalid choice of basis points!")

    return basis


def compute(basis: np.array, num_dilates: int, mode: str, **kwargs):
    polytope = Polytope(chosen_basis, num_dilates=num_dilates)

    # find all possible combinations of basis vectors
    polytope.compute_permutations()
    polytope.get_unique_permutations()

    if kwargs.get("show", True):
        polytope.show(**kwargs)

    return polytope


if __name__ == "__main__":
    config = ConfigParser()
    config.read(r"../config.ini")
    dilates = [int(d) for d in config["SETTINGS"]["num_dilates"].split(",")]

    # get basis points
    chosen_basis = choose_basis(config)
    chosen_basis = [Point(point, mode=config["SETTINGS"]["mode"]) for point in chosen_basis]

    polys = []
    for idx, d in enumerate(dilates):
        poly = compute(chosen_basis, d, config["SETTINGS"]["mode"],
                       show=config["SETTINGS"].getboolean("show_individuals"))
        poly.coloring = np.full((poly.num_unique_permutations,), idx)
        polys.append(poly)

    if config["SETTINGS"].getboolean("show_progression"):
        total_poly = JoinedPolytope(*polys)
        total_poly.show()
