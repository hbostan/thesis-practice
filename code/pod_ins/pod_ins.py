import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from green_gauss_theorem.vec2 import Vec2
from green_gauss_theorem.triangle import Triangle, Inflow, Outflow
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def plot(elements, mesh_Vx, mesh_Vy, variable):
    # variable = np.array(variable)/np.max(np.abs(variable))
    fig, ax = plt.subplots()
    tri_buf = []
    for t in elements:
        tpatch = Polygon(t.get_vertex_coords(), fill=False)
        tri_buf.append(tpatch)
    tri_buf = PatchCollection(tri_buf, linewidths=0.1, edgecolors='black')
    tri_buf.set_array(variable)
    variable = np.array(variable)
    tri_buf.set_cmap('jet')
    ax.add_collection(tri_buf)
    ax.set_ylim(np.min(mesh_Vy), np.max(mesh_Vy))
    ax.set_xlim(np.min(mesh_Vx), np.max(mesh_Vx))
    ax.set_box_aspect((np.max(mesh_Vx)-np.min(mesh_Vx))/(np.max(mesh_Vy)-np.min(mesh_Vy)))
    fig.colorbar(tri_buf)
    print(variable.max())
    print(variable.min())
    # plt.show()

def calculate_first_derivative(elements, u, v, p):
    u_first_derivative_x = []
    u_first_derivative_y = []
    v_first_derivative_x = []
    v_first_derivative_y = []
    for i,tri in enumerate(elements):
        # update : calculates corresponding u, v and p at cell centers. 
        tri.update(u[i], v[i], p[i])
    for tri in elements:
        u_first_derivative = tri.u_find_first_derivative()
        v_first_derivative = tri.v_find_first_derivative()

        dudx = u_first_derivative.x
        dudy = u_first_derivative.y
        u_first_derivative_x.append(dudx)
        u_first_derivative_y.append(dudy)

        dvdx = v_first_derivative.x
        dvdy = v_first_derivative.y
        v_first_derivative_x.append(dvdx)
        v_first_derivative_y.append(dvdy)

    return u_first_derivative_x, u_first_derivative_y, v_first_derivative_x, v_first_derivative_y

def calculate_second_derivative(elements, u, v, p):
    u_second_derivative_x = []
    u_second_derivative_y = []

    v_second_derivative_x = []
    v_second_derivative_y = []

    for i,tri in enumerate(elements):
        # update : calculates corresponding u, v and p at cell centers. 
        tri.update(u[i], v[i], p[i])
    for tri in elements:
        u_second_derivative = tri.u_find_second_derivative()
        v_second_derivative = tri.v_find_second_derivative()

        dudxx = u_second_derivative.x
        dudyy = u_second_derivative.y
        u_second_derivative_x.append(dudxx)
        u_second_derivative_y.append(dudyy)

        dvdxx = v_second_derivative.x
        dvdyy = v_second_derivative.y
        v_second_derivative_x.append(dvdxx)
        v_second_derivative_y.append(dvdyy)

    return u_second_derivative_x, u_second_derivative_y, v_second_derivative_x, v_second_derivative_y

# ----------------------------------------- RESULTS READ ----------------------------------------- #
plot_check = True

TimeStep_DGNu = [] 
TimeStep_DGNv = [] 
TimeStep_DGNp = [] 

mesh_contents = open('/home/damla/Desktop/Thesis/code/pod_ins/mesh/mesh_info.txt', 'r').read()
exec(mesh_contents)

DIR = '/home/damla/Desktop/Thesis/code/pod_ins/practice_results'
for file_name in sorted(os.listdir(DIR)):
    file_name = os.path.join(DIR, file_name)
    result_contents = open(file_name, 'r').read()
    exec(result_contents)

    TimeStep_DGNu.append(np.array(DGNu))
    TimeStep_DGNv.append(np.array(DGNv))
    TimeStep_DGNp.append(np.array(DGNp))
    del result_contents

# ------------------------------------------ PARAMETERS ------------------------------------------ #

NUM_TIME_STEPS    = len(TimeStep_DGNu)
NUM_ELEMENTS      = len(TimeStep_DGNu[0])
NODES_PER_ELEMENT = len(TimeStep_DGNu[0][0])
NUM_NODES         = NUM_ELEMENTS * NODES_PER_ELEMENT
NUM_POD_MODES     = 1

U_INFLOW = 1
V_INFLOW = 0
P_INFLOW = 0

x_coord = np.array(mesh_DGNx).flatten()
y_coord = np.array(mesh_DGNy).flatten()

U = np.zeros((NUM_TIME_STEPS, NUM_NODES))
V = np.zeros((NUM_TIME_STEPS, NUM_NODES))
P = np.zeros((NUM_TIME_STEPS, NUM_NODES))

# --------------------------------------------- MESH --------------------------------------------- #
# elements [] stores tiangle mesh elements 
elements:list[Triangle] = []       
for e in range(NUM_ELEMENTS):
    # triangle mesh elements requires vertex & node coordinates.
    element = Triangle(mesh_Vx[e], mesh_Vy[e], mesh_DGNx[e], mesh_DGNy[e])
    elements.append(element) 

# Fill in neighbors of each triangle
for elem_idx, neighbor_list in enumerate(EToE):
# for elem_idx in range(len(EToE)):
#     neighbor_list = EToE[elem_idx]
    for n_idx, n in enumerate(neighbor_list):
        if n != -1:
            neighbor = elements[n]
            elements[elem_idx].add_neighbor(neighbor)
        else:
            # We have a boundary
            boundary_type = EToB[elem_idx][n_idx]
            if boundary_type == 1:
                elements[elem_idx].add_neighbor(None)
            elif boundary_type == 2:
                center = elements[elem_idx].edge_centers[n_idx]
                inflow = Inflow(center, elements[elem_idx], U_INFLOW, V_INFLOW, P_INFLOW)
                elements[elem_idx].add_neighbor(inflow)
            elif boundary_type == 3:
                center = elements[elem_idx].edge_centers[n_idx]
                outflow = Outflow(center, elements[elem_idx])
                elements[elem_idx].add_neighbor(outflow)
            else:
                elements[elem_idx].add_neighbor(None)

# --------------------------------------------- POD --------------------------------------------- #
for i in range(NUM_TIME_STEPS):
    U[i, :] = TimeStep_DGNu[i].flatten()
    V[i, :] = TimeStep_DGNv[i].flatten()
    P[i, :] = TimeStep_DGNp[i].flatten()

uMean = np.mean(U, axis=0)
vMean = np.mean(V, axis=0)
pMean = np.mean(P, axis=0)

uFluc = np.zeros((NUM_TIME_STEPS, NUM_NODES))
vFluc = np.zeros((NUM_TIME_STEPS, NUM_NODES))

for t in range(NUM_TIME_STEPS):
    uFluc[t, :] = U[t, :] - uMean
    vFluc[t, :] = V[t, :] - vMean

SNAPSHOT = np.concatenate((uFluc, vFluc), axis=1)
covariance = np.cov(SNAPSHOT)

# Eigenvalues and eigenvector of correlation matrix.
eig_vals, eig_vecs = LA.eig(covariance)
eig_vals = np.real(eig_vals)
eig_vecs = np.real(eig_vecs)

# Sort eigenvalues and vectors according to abs(eig_val)
sorting_indices = np.abs(eig_vals).argsort()[::-1]
eig_vals = eig_vals[sorting_indices]
eig_vecs = eig_vecs[:, sorting_indices]

# Scale Time Coefficients
TimeCoeff = np.zeros((NUM_TIME_STEPS, NUM_POD_MODES))

for i in range(NUM_POD_MODES):
    for j in range(NUM_TIME_STEPS):
        ModeFactor = np.sqrt(NUM_TIME_STEPS * eig_vals[i])
        TimeCoeff[j][i] = eig_vecs[j][i] * ModeFactor

# Calculate POD modes
uPOD = np.zeros((NUM_POD_MODES, NUM_NODES))
vPOD = np.zeros((NUM_POD_MODES, NUM_NODES))

for i in range(NUM_POD_MODES):
    for j in range(NUM_TIME_STEPS):
        uPOD[i, :] = uPOD[i, :] + eig_vecs[j][i] * uFluc[j, :]
        vPOD[i, :] = vPOD[i, :] + eig_vecs[j][i] * vFluc[j, :]

    modeFactor = 1 / np.sqrt(NUM_TIME_STEPS*eig_vals[i])
    uPOD[i, :] = uPOD[i, :]*modeFactor
    vPOD[i, :] = vPOD[i, :]*modeFactor

# ------------------------------------------ DERIVATIVES ------------------------------------------#
uPOD  = uPOD.reshape(NUM_POD_MODES, NUM_ELEMENTS, NODES_PER_ELEMENT)
vPOD  = vPOD.reshape(NUM_POD_MODES, NUM_ELEMENTS, NODES_PER_ELEMENT)
uMean = uMean.reshape(NUM_ELEMENTS, NODES_PER_ELEMENT)
vMean = vMean.reshape(NUM_ELEMENTS, NODES_PER_ELEMENT)
p = TimeStep_DGNp[-1]

uPOD_first_derivative_x = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))
uPOD_first_derivative_y = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))
vPOD_first_derivative_x = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))
vPOD_first_derivative_y = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))

uPOD_second_derivative_x = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))
uPOD_second_derivative_y = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))
vPOD_second_derivative_x = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))
vPOD_second_derivative_y = np.zeros((NUM_POD_MODES, NUM_ELEMENTS))

uMean_first_derivative_x, uMean_first_derivative_y, vMean_first_derivative_x, vMean_first_derivative_y = calculate_first_derivative(elements, uMean, vMean, p)
uMean_second_derivative_x, uMean_second_derivative_y, vMean_second_derivative_x, vMean_second_derivative_y = calculate_second_derivative(elements, uMean, vMean, p)

# POD MODE DERIVATIVES
for i in range(NUM_POD_MODES):
    u = uPOD[i, :, :]
    v = vPOD[i, :, :]

    u_first_derivative_x, u_first_derivative_y, v_first_derivative_x, v_first_derivative_y = calculate_first_derivative(elements, u, v, p)
    u_second_derivative_x, u_second_derivative_y, v_second_derivative_x, v_second_derivative_y = calculate_second_derivative(elements, u, v, p)
    
    uPOD_first_derivative_x[i, :] = u_first_derivative_x
    uPOD_first_derivative_y[i, :] = u_first_derivative_y
    vPOD_first_derivative_x[i, :] = v_first_derivative_x
    vPOD_first_derivative_y[i, :] = v_first_derivative_y

    uPOD_second_derivative_x[i, :] = u_second_derivative_x
    uPOD_second_derivative_y[i, :] = u_second_derivative_y
    vPOD_second_derivative_x[i, :] = v_second_derivative_x
    vPOD_second_derivative_y[i, :] = v_second_derivative_y

# --------------------------------------------- PLOTS ---------------------------------------------#

font = {
    'family': 'serif',
    'color': 'black',
    'weight': 'normal',
    'size': 18,
}

if plot_check:

    variable = []
    variable2 = []
    for j,tri in enumerate(elements):
        tri.update(TimeStep_DGNu[-1][j], TimeStep_DGNv[-1][j], TimeStep_DGNp[-1][j])
    for tri in elements:
        var = tri.u_find_second_derivative()
        var2 = tri.u_find_first_derivative()
        variable.append(var.x)
        variable2.append(var2.x)
    plot(elements, mesh_Vx, mesh_Vy, variable)
    plot(elements, mesh_Vx, mesh_Vy, variable2)
    plt.show()
    

    # for i in range(NUM_POD_MODES):
    #     uPOD_ = uPOD[i, :, :]
    #     vPOD_ = vPOD[i, :, :]
    #     variable = []
    #     for j,tri in enumerate(elements):
    #         tri.update(uPOD_[j], vPOD_[j], TimeStep_DGNp[-1][j])
    #     for tri in elements:
    #         var = tri.u_find_first_derivative()
    #         variable.append(var.x)
    #     plot(elements, mesh_Vx, mesh_Vy, variable)


    # relative_importance_content = np.zeros(NUM_TIME_STEPS)
    # total_energy = np.sum(eig_vals)
    # acc = 0
    # for i in range(len(eig_vals)):
    #     acc = acc + eig_vals[i]
    #     relative_importance_content[i] = acc / total_energy * 100

    #     #  Plot first n_modes important eigenvalues

    # figure, axis = plt.subplots(1, 2)
    # axis[0].plot(np.arange(1, NUM_POD_MODES + 1), eig_vals[:NUM_POD_MODES], '-o', color='black', ms=8, alpha=1, mfc='red')
    # axis[0].set_xlabel('k', fontdict=font)
    # axis[0].set_ylabel(r'$λ_{k}$', fontdict=font)
    # axis[0].grid()

    # axis[1].plot(np.arange(1, NUM_POD_MODES + 1),
    #             relative_importance_content[:NUM_POD_MODES],
    #             '-o',
    #             color='black',
    #             ms=8,
    #             alpha=1,
    #             mfc='red')
    # axis[1].set_xlabel('k', fontdict=font)
    # axis[1].set_ylabel(r'$RIC_{k}(\%)$', fontdict=font)
    # axis[1].grid()
    # # plt.savefig('Relative Importance Content.png')
    # plt.show()    



