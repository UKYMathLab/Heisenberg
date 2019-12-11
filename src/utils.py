import numpy as np
import pptk
from pathlib import Path

def point_cloud(points, size: float):

    v = pptk.viewer(points)
    v.attributes(points[:, 2])
    # v.attributes(np.sum(points[:, :],axis=1))
    v.set(point_size=size, floor_level=0)

    v.set(bg_color=[1,1,1,1], show_info=False, show_axis=False)
    v.color_map("summer")
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
    v.play(poses, repeat=True)
    # save_dir = Path().cwd() / "recordings"
    # v.record(folder=save_dir, poses=poses, ts=0.5*np.arange(len(poses)), fps=30)
    v.wait()
