import numpy as np
import math
import os
from vec2 import Vec2
from triangle import Triangle, Inflow, Outflow
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import SymLogNorm

DIR='/home/damla/Desktop/POD_INS/TXT_TRIAL'

f_contents = open('/home/damla/Desktop/POD_INS/mesh_info.txt', 'r').read()
exec(f_contents)

NUM_ELEMENTS = len(EToE)
NODES_PER_ELEMENT = 15
U_INFLOW = 1
V_INFLOW = 0
P_INFLOW = 0

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
            

# TimeStep_DGNu = [] # TimeStep_DGNu[time_Step][element][node]
# TimeStep_DGNv = [] # TimeStep_DGNv[time_Step][element][node]
# TimeStep_DGNp = [] # TimeStep_DGNp[time_Step][element][node]
colors = []

# FILE READ
for file_name in sorted(os.listdir(DIR)):
    file_name = os.path.join(DIR, file_name)
    f_contents = open(file_name, 'r').read()
    exec(f_contents)
    for i,tri in enumerate(elements):
      # .update : calculates corresponding u, v and p at cell centers. 
      tri.update(DGNu[i], DGNv[i], DGNp[i])
    for tri in elements:
        u_dd = tri.u_find_second_derivative()
        colors.append(u_dd.y)
    ## Do whatever 
    del f_contents

colors = np.array(colors)/np.max(np.abs(colors))
fig, ax = plt.subplots()
tri_buf = []
for t in elements:
    tpatch = Polygon(t.get_vertex_coords(), fill=False)
    tri_buf.append(tpatch)
tri_buf = PatchCollection(tri_buf, linewidths=0.1, edgecolors='black')
tri_buf.set_array(colors)
colors = np.array(colors)
tri_buf.set_cmap('jet')
ax.add_collection(tri_buf)
ax.set_ylim(np.min(mesh_Vy), np.max(mesh_Vy))
ax.set_xlim(np.min(mesh_Vx), np.max(mesh_Vx))
ax.set_box_aspect((np.max(mesh_Vx)-np.min(mesh_Vx))/(np.max(mesh_Vy)-np.min(mesh_Vy)))
fig.colorbar(tri_buf)
plt.show()



