import numpy as np
import pptk

def point_cloud(points, size: float):

    v = pptk.viewer(points)
    v.attributes(points[:, 2])
    v.set(point_size=size, floor_level=0)
    v.wait()

