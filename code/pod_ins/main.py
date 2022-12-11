import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from topology.mesh import Mesh
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import fine_mesh.mesh_info as mesh_info
from scipy import integrate
import concurrent.futures
from utils.result_reader import read_from_dir
from pod import calculate_pod_modes
from galerkin_projection import galerkin_projection


# Plots a mesh using `variable` as the values for coloring.
def plot_mesh(mesh, variable):
    fig, ax = plt.subplots()
    tri_buf = []
    for t in mesh.triangles:
        tpatch = Polygon(t.get_vertex_coords(), fill=False)
        tri_buf.append(tpatch)
    tri_buf = PatchCollection(tri_buf, linewidths=0.1, edgecolors='black')
    tri_buf.set_array(variable)
    variable = np.array(variable)
    tri_buf.set_cmap('jet')
    ax.add_collection(tri_buf)
    bbox = mesh.bbox()
    ax.set_ylim(bbox[1])
    ax.set_xlim(bbox[0])
    ax.set_box_aspect((bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
    fig.colorbar(tri_buf)


# ----------------------------------------- RESULTS READ ----------------------------------------- #

DIR = 'fine_re200'
TimeStep_DGNu, TimeStep_DGNv, TimeStep_DGNp, num_ts = read_from_dir(DIR)
mesh = Mesh(mesh_info)
# ------------------------------------------ PARAMETERS ------------------------------------------ #

NUM_TIME_STEPS = num_ts
NUM_ELEMENTS = mesh.num_triangles
NODES_PER_ELEMENT = mesh.num_node_per_triangle
NUM_NODES = NUM_ELEMENTS * NODES_PER_ELEMENT
NUM_POD_MODES = 10

U_INFLOW = 1
V_INFLOW = 0
P_INFLOW = 0
RE = 200
VISC = 1 / RE
FINAL_T = 100
TIME_STEP = 1
TIME = np.linspace(0, FINAL_T, NUM_TIME_STEPS)

pod_result = calculate_pod_modes(TimeStep_DGNu, TimeStep_DGNv, TimeStep_DGNp, NUM_TIME_STEPS, NUM_NODES, NUM_POD_MODES)
TimeCoeffGalerkin, time_galerkin = galerkin_projection(mesh, pod_result, VISC, FINAL_T, TIME_STEP)

# u_reconstruct = np.zeros((NUM_TIME_STEPS, NUM_ELEMENTS))
# v_reconstruct = np.zeros((NUM_TIME_STEPS, NUM_ELEMENTS))

# for time in range(NUM_TIME_STEPS):
#     for mode in range(NUM_POD_MODES):
#         u_reconstruct[time, :] += pod_result.time_coeff[time, mode] * pod_result.uPOD[mode, :]
#         v_reconstruct[time, :] += pod_result.time_coeff[time, mode] * pod_result.vPOD[mode, :]
#     u_reconstruct[time, :] += pod_result.uMean
#     v_reconstruct[time, :] += pod_result.vMean

# u_galerkin = np.zeros((NUM_TIME_STEPS, NUM_ELEMENTS))
# v_galerkin = np.zeros((NUM_TIME_STEPS, NUM_ELEMENTS))

# for time in range(NUM_TIME_STEPS):
#     for mode in range(NUM_POD_MODES):
#         u_galerkin[time, :] += TimeCoeffGalerkin[time, mode] * pod_result.uPOD[mode, :]
#         v_galerkin[time, :] += TimeCoeffGalerkin[time, mode] * pod_result.vPOD[mode, :]
#     u_galerkin[time, :] += pod_result.uMean
#     v_galerkin[time, :] += pod_result.vMean

# plot_mesh(mesh, u_galerkin[-1, :])
# plt.show()

# --------------------------------------------- PLOTS ---------------------------------------------#
font = {
    'family': 'serif',
    'color': 'black',
    'weight': 'normal',
    'size': 18,
}

# colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
lines = []
for mode in range(1):
    l, = plt.plot(TIME,
                  pod_result.time_coeff[:, mode],
                  linestyle='-',
                  marker='+',
                  linewidth=4,
                  markersize=10,
                  alpha=0.8,
                  label=rf'$a_{mode}{{t}}$ - True')
    l2, = plt.plot(time_galerkin,
                   TimeCoeffGalerkin[:, mode],
                   linestyle='--',
                   marker='o',
                   linewidth=4,
                   markersize=5,
                   alpha=0.8,
                   label=rf'$a_{mode}{{t}}$ - GP')
    plt.xlabel("Time", fontdict=font)
    plt.ylabel("Time coefficients", fontdict=font)
    lines.append(l)
    lines.append(l2)
plt.legend(handles=lines)
plt.show()

# variable = []
# for j, tri in enumerate(mesh.triangles):
#     tri.update(TimeStep_DGNu[-1][j], TimeStep_DGNv[-1][j], TimeStep_DGNp[-1][j])
# for tri in mesh.triangles:
#     var = tri.u_at_center
#     variable.append(var)
# plot_mesh(mesh, variable)
# plt.show()

# for i in range(NUM_POD_MODES):
#     uPOD_ = uPOD[i, :, :]
#     vPOD_ = vPOD[i, :, :]
#     variable = []
#     for j,tri in enumerate(mesh.triangles):
#         tri.update(uPOD_[j], vPOD_[j], TimeStep_DGNp[-1][j])
#     for tri in mesh.triangles:
#         var = tri.u_at_center
#         variable.append(var)
#     plot_mesh(mesh, variable)
#     plt.show()
