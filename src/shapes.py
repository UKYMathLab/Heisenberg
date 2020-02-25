from pathlib import Path
from math import factorial
import itertools as it
from typing import Iterable

import numpy as np
import pptk
from tqdm import tqdm

import drivers


class Point:
    def __init__(self, point: np.ndarray, mode: str = "h"):
        # catch errors
        if mode not in {"n", "h", "d"}:
            raise NotImplementedError(f"Mode {mode} is not valid!")
        if point.ndim != 1:
            raise ValueError("Point class only supports 1-dimensional arrays.")

        self.coordinates = point
        self.mode = mode

    def __repr__(self):
        return f"{self.__class__.__name__}(point={self.point})"

    def __add__(self, other):
        normal_sum = self.coordinates + other.coordinates

        # Heisenberg twist
        if self.mode == "h":
            twisted_z = self.x * other.y
            normal_sum[2] += twisted_z

        # Heisenberg Moon Duchin twist
        elif self.mode == "d":
            twisted_z = (self.x*other.y - self.y*other.x)/2
            normal_sum[2] += twisted_z

        return Point(normal_sum, mode=self.mode)

    def __radd__(self, other):
        other.__add__(self)

    @property
    def num_dimensions(self):
        return self.coordinates.ndim

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[2]


class Polytope:
    r"""A collection of points that are the discrete realization of
    a polytope in the normal sense. In our Heisenberg setting, a
    polytope S_k is the realization of the k-fold product of points
    in a generating set S.
    """

    def __init__(self, basis_points: Iterable, num_dilates: int):
        self.basis_points = basis_points
        self.num_dilates = num_dilates
        self.mode = self.basis_points[0].mode

        self.all_permutations = []
        self.unique_permutations = np.zeros((1, 3))

        self.coloring = None

    def __add__(self, other):
        # check that polytopes match in modes
        if self.mode != other.mode:
            raise ValueError(f"The modes of {self} and {other} do not match!")

        # combine unique permutations with coloring and then combine across polytopes
        for p in {self, other}:
            # check that polytopes have actually been computed
            if not any(p.unique_permutations):
                p.get_unique_permutations()

            # make sure there are identifiers for points from each polytope
            if p.coloring is None:
                p.coloring = np.full((p.num_unique_permutations,), 0)

        return JoinedPolytope(self, other)


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
        self.all_permutations = list(tqdm(it.product(self.basis_points, repeat=self.num_dilates),
                                          total=len(self.basis_points)**self.num_dilates,
                                          desc=f"Computing dilate n={self.num_dilates}"))

        # check computations went correctly
        drivers.CheckValidLength(len(self.all_permutations),
                                 len(self.basis_points)**self.num_dilates,
                                 mode="equal")

        return self.all_permutations

    def get_unique_permutations(self):
        r"""Finds all unique elements of an array."""

        # sum all permutations and get unique points
        results = []
        for permutation in tqdm(self.all_permutations,
                                total=self.num_permutations,
                                desc=f"Evaluating dilate n={self.num_dilates}"):
            intermediate_result = None
            for p in permutation:
                if intermediate_result is None:
                    intermediate_result = p
                else:
                    intermediate_result += p
            results.append(intermediate_result)
        self.unique_permutations = np.unique(np.vstack([p.coordinates for p in results]),
                                             axis=0)

        # check computations went correctly
        drivers.CheckValidLength(self.num_unique_permutations, self.num_permutations,
                                 mode="less than", strict=False)

    def show(self, **kwargs):
        r"""Plots the points of the polytope as a point cloud."""

        v = pptk.viewer(self.unique_permutations)
        v.attributes(kwargs.get("attributes", self.unique_permutations[:, 2]))
        # v.attributes(np.sum(points[:, :],axis=1))
        v.set(point_size=kwargs.get("point_size", 0.25),
              floor_level=kwargs.get("floor_level", 0),
              bg_color=kwargs.get("bg_color", [1,1,1,1]),
              show_info=kwargs.get("show_info", False),
              show_axis=kwargs.get("show_axis", False))
        v.color_map(kwargs.get("color_map", "summer"))

        if kwargs.get("play", False) or kwargs.get("record", False):
            # [x, y, z, phi, theta, r]
            n = 10
            x,y = 0,0
            theta = np.pi/6
            r = 70
            poses = [[x,y,0, 0*np.pi/2, theta, r],
                     [x,y,0, 1*np.pi/2, theta, r],
                     [x,y,0, 2*np.pi/2, theta, r],
                     [x,y,0, 3*np.pi/2, theta, r],
                     [x,y,0, 4*np.pi/2, theta, r]]

            # rotate point cloud
            if kwargs.get("play", False):
                v.play(poses, repeat=kwargs.get("repeat", False))

            # record rotating point cloud
            if kwargs.get("record", False):
                save_dir = kwargs.get("save_dir", Path().cwd()/"recordings")
                v.record(folder=save_dir, poses=poses,
                         ts=0.5*np.arange(len(poses)),
                         fps=kwargs.get("fps", 30))

            # pause execution of script until ENTER key is pressed
            if kwargs.get("wait", False):
                v.wait()


class JoinedPolytope:
    def __init__(self, *args):
        r"""
        Arguments passed in should be Polytope classes.
        """

        self.polytopes = [p for p in args]

        # probably need to keep as list comprehensions so order is preserved
        self.points = np.vstack([p.unique_permutations for p in self.polytopes])
        self.coloring = np.hstack([p.coloring for p in self.polytopes])
        print(self.points.shape)
        print(self.coloring.shape)

    def show(self, **kwargs):
        r"""Plots the points of the polytope as a point cloud."""

        v = pptk.viewer(self.points)
        v.attributes(kwargs.get("attributes", self.coloring))
        # v.attributes(np.sum(points[:, :],axis=1))
        v.set(point_size=kwargs.get("point_size", 0.25),
              floor_level=kwargs.get("floor_level", 0),
              bg_color=kwargs.get("bg_color", [1,1,1,1]),
              show_info=kwargs.get("show_info", False),
              show_axis=kwargs.get("show_axis", False))
        v.color_map(kwargs.get("color_map", "summer"))

        if kwargs.get("play", False) or kwargs.get("record", False):
            # [x, y, z, phi, theta, r]
            n = 10
            x,y = n/2,n/2
            theta = np.pi/6
            r = 35
            poses = [[x,y,0, 0*np.pi/2, theta, r],
                     [x,y,0, 1*np.pi/2, theta, r],
                     [x,y,0, 2*np.pi/2, theta, r],
                     [x,y,0, 3*np.pi/2, theta, r],
                     [x,y,0, 4*np.pi/2, theta, r]]

            # rotate point cloud
            if kwargs.get("play", False):
                v.play(poses, repeat=kwargs.get("repeat", False))

            # record rotating point cloud
            if kwargs.get("record", False):
                save_dir = kwargs.get("save_dir", Path().cwd()/"recordings")
                v.record(folder=save_dir, poses=poses,
                         ts=0.5*np.arange(len(poses)),
                         fps=kwargs.get("fps", 30))

            # pause execution of script until ENTER key is pressed
            if kwargs.get("wait", False):
                v.wait()


def _permutation_sum(permutation: np.array) -> np.array:

    # ExamineData(permutation, "Permutation")
    max_iter = permutation.shape[0] - 1
    # check computations went correctly
    drivers.CheckValidLength(max_iter, 0, mode="greater than")

    for i  in range(max_iter):
        # make sure in valid range (since operating on pairs)
        intermediate_result = permutation[i]+permutation[i+1]
        permutation[i+1] = intermediate_result
    # drivers.ExamineData(intermediate_result, "intermediate").pause()

    return intermediate_result
