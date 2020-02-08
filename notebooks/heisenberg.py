from typing import List, Iterable
import itertools as it

import numpy as np

Vec3 = np.ndarray


class PlotForm:
    xs: List[float]
    ys: List[float]
    zs: List[float]

    def __init__(self, xs, ys, zs):
        self.xs = xs
        self.ys = ys
        self.zs = zs

    @staticmethod
    def from_pt_set(pts: List[Vec3]) -> 'PlotForm':
        lists = ([], [], [])
        for pt in pts:
            for (element, list_) in zip(pt, lists):
                list_.append(element)
        xs, ys, zs = lists
        return PlotForm(xs, ys, zs)

    def plotme(self, fig, **kwargs):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.xs, self.ys, self.zs, **kwargs)


def heisen3_sum(v1: Vec3, v2: Vec3) -> Vec3:
    """
    Add two 3-dimensional vectors, Heisenberg style.
    """
    assert v1.shape == v2.shape and v1.shape == (3,)
    # first, do normal sum to get (x + x', y + y', z + z')
    out = v1 + v2
    # then add the xy' part to z + z'
    out[2] += v1[0] * v2[1]
    return out


def vectuple_h3_sum(vectuple: Iterable[Vec3]) -> Vec3:
    """
    Heisenberg-sum an arbitrary number of vectors.
    """
    out = np.array([0, 0, 0])
    for t in vectuple:
        out = heisen3_sum(out, t)
    return np.array(out)


def compute_h3_pn(S: List[Vec3], n: int) -> Iterable[Vec3]:
    """
    Given a finite set of points S and an integer n, compute the set P_n, where
    P_n is

    P_n := {a1 * a2 * ... * an | a_i in S},

    where multiplication is the Heisenberg sum.

    Note that points may appear more than once in the sequence generated.
    """
    # It just occurred to me that we could speed up the computation of P_n for
    # generating sets containing 0 by computing the nth dilate iteratively on S
    # \ {(0, 0, 0)}.
    #
    # We could maybe generalize this to any set where there's some (0, 0, z)
    # vectors? Compute the... hm. I think we can do this. Let S' be the
    # generating set with all the z-vectors removed. Now... Compute the dilates
    # P_k(S') for k = 1, 2, ..., n. For k = 1, ..., n - 1, shift up P_k(S') by
    # z_1 + ... + z_(n - k) for some nonzero z values that you can get in S.
    #
    # This might not be that much faster if you don't have zero. Ugh. Also I
    # might have written it down wrong. Or thought wrong. Ugh.
    #
    # Instead of having to compute products for |S|^n things, you instead have
    # to compute products for (|S| - 1) + (|S| - 1)^2 + ... + (|S| - 1)^n
    # things. If you have c of the z vectors then... I think would be
    # (|S| - 1) + (|S| - 1)^2 + ... + (|S| - c)^n products
    assert n >= 0
    if n == 0:
        return set()
    tuples = it.product(S, repeat=n)
    p_n = map(vectuple_h3_sum, tuples)
    return p_n
