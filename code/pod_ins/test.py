import meshio
import matplotlib.pyplot as plt
from geometry.mesh.cartesian_mesh import CartesianStructuredMesh
import utils.plotting as cp
import numpy as np
import os
from scipy.interpolate import griddata

meshio_mesh = meshio.read('test.vtu')
cartesian_mesh = CartesianStructuredMesh(meshio_mesh)

xs = [n.x for n in cartesian_mesh.nodes]
ys = [n.y for n in cartesian_mesh.nodes]
val = [n.u_value for n in cartesian_mesh.nodes]
dx, dy = cartesian_mesh.finite_differences(data=val, compute_laplacian=False)
# plt.scatter(xs, ys, c=val)
fig = None
gx, gy, gd = cp.plot_square_data(xs, ys, dx)
plt.clf()
plt.cla()
from matplotlib.animation import PillowWriter

writer = PillowWriter()
fig, ax = plt.subplots(figsize=(4, 4))


def func(path):
    cartesian_mesh = CartesianStructuredMesh(meshio.read(path))
    x = [n.x for n in cartesian_mesh.nodes]
    y = [n.y for n in cartesian_mesh.nodes]
    u_val = [n.u_value for n in cartesian_mesh.nodes]
    xmin = np.round(np.min(x), 1)
    xmax = np.round(np.max(x), 1)
    ymin = np.round(np.min(y), 1)
    ymax = np.round(np.max(y), 1)
    dx = (xmax - xmin) / 300
    dy = (ymax - ymin) / 300
    gridx = np.arange(xmin, xmax, dx)
    gridy = np.arange(ymin, ymax, dy)
    gridx, gridy = np.meshgrid(gridx, gridy)
    # interpolate data
    grid = griddata(np.column_stack((x, y)), u_val, (gridx, gridy))
    return gridx, gridy, grid


print('Giffing')
with writer.saving(fig, 'test.gif', 100):
    dir = 'results/sqr_re100_ts_025'
    for fname in sorted(os.listdir(dir))[-20:]:
        print(fname)
        gx, gy, gd = func(os.path.join(dir, fname))
        ax.contourf(gx, gy, gd)
        writer.grab_frame()
        plt.cla()
