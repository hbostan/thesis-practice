import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


class NODE:
    def __init__(self, x_coord, y_coord, u, v, p):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.u = u
        self.v = v
        self.p = p
    
    def __eq__(self, other):
        if self.x_coord == other.x_coord and self.y_coord == other.y_coord:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

plot_check = False

TimeStep_DGNu = [] # TimeStep_DGNu[time_Step][element][node]
TimeStep_DGNv = [] # TimeStep_DGNv[time_Step][element][node]
TimeStep_DGNp = [] # TimeStep_DGNp[time_Step][element][node]

f_contents = open('mesh_info.txt', 'r').read()
exec(f_contents)

NUM_ELEMENTS = len(EToE)
NODES_PER_ELEMENT = 15

# FILE READ
DIR = 'TXTv3'
for file_name in sorted(os.listdir(DIR)):
    file_name = os.path.join(DIR, file_name)
    f_contents = open(file_name, 'r').read()
    exec(f_contents)

    TimeStep_DGNu.append(np.array(DGNu))
    TimeStep_DGNv.append(np.array(DGNv))
    TimeStep_DGNp.append(np.array(DGNp))
    del f_contents

uMean = np.mean(TimeStep_DGNu, axis=0)
with open('umean.pkl', 'wb') as f:
    pickle.dump(uMean, f)
print('Done!')



NUM_TIME_STEPS = len(TimeStep_DGNu)

TimeStep_NODEs = [] # TimeStep_DGNp[time_Step][node]

for t in range(NUM_TIME_STEPS):

    NODEs = []
    process_coords = set()

    for e in range(NUM_ELEMENTS):
        for n in range(NODES_PER_ELEMENT):
            node = NODE(mesh_DGNx[e][n], mesh_DGNy[e][n], TimeStep_DGNu[t][e][n], TimeStep_DGNv[t][e][n], TimeStep_DGNp[t][e][n])
            if not (node.x_coord, node.y_coord) in process_coords: # if not node in NODEs:
                NODEs.append(node)
                process_coords.add((node.x_coord, node.y_coord))

    TimeStep_NODEs.append(NODEs)


NUM_NODES = len(NODEs)
print(NUM_ELEMENTS)

x_coord = np.zeros(NUM_NODES)
y_coord = np.zeros(NUM_NODES)

for n in range(NUM_NODES):
    x_coord[n] = NODEs[n].x_coord
    y_coord[n] = NODEs[n].y_coord

TimeStep_u = np.zeros((NUM_TIME_STEPS, NUM_NODES))
TimeStep_v = np.zeros((NUM_TIME_STEPS, NUM_NODES))
TimeStep_p = np.zeros((NUM_TIME_STEPS, NUM_NODES))

for t in range(NUM_TIME_STEPS):
    u = np.zeros(NUM_NODES)
    v = np.zeros(NUM_NODES)
    p = np.zeros(NUM_NODES)
    for n in range(NUM_NODES):
        u[n] = TimeStep_NODEs[t][n].u
        v[n] = TimeStep_NODEs[t][n].v
        p[n] = TimeStep_NODEs[t][n].p

    TimeStep_u[t, :] = u
    TimeStep_v[t, :] = v
    TimeStep_p[t, :] = p

uMean = np.mean(TimeStep_u, axis=0)
vMean = np.mean(TimeStep_v, axis=0)
pMean = np.mean(TimeStep_p, axis=0)

uFluc = np.zeros((NUM_TIME_STEPS, NUM_NODES))
vFluc = np.zeros((NUM_TIME_STEPS, NUM_NODES))

for t in range(NUM_TIME_STEPS):
    uFluc[t, :] = TimeStep_u[t, :] - uMean
    vFluc[t, :] = TimeStep_v[t, :] - vMean

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

NUM_POD_MODES = 6

# Scale Time Coefficients
TimeCoeff = np.zeros((NUM_TIME_STEPS, NUM_POD_MODES))

for i in range(NUM_POD_MODES):
    for j in range(NUM_TIME_STEPS):
        ModeFactor = np.sqrt(NUM_TIME_STEPS*eig_vals[i])
        TimeCoeff[j][i] = eig_vecs[j][i]*ModeFactor

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

font = {
    'family': 'serif',
    'color': 'black',
    'weight': 'normal',
    'size': 18,
}

if plot_check:
    relative_importance_content = np.zeros(NUM_TIME_STEPS)
    total_energy = np.sum(eig_vals)
    acc = 0
    for i in range(len(eig_vals)):
        acc = acc + eig_vals[i]
        relative_importance_content[i] = acc / total_energy * 100

        #  Plot first n_modes important eigenvalues

    figure, axis = plt.subplots(1, 2)
    axis[0].plot(np.arange(1, NUM_POD_MODES + 1), eig_vals[:NUM_POD_MODES], '-o', color='black', ms=8, alpha=1, mfc='red')
    axis[0].set_xlabel('k', fontdict=font)
    axis[0].set_ylabel(r'$λ_{k}$', fontdict=font)
    axis[0].grid()

    axis[1].plot(np.arange(1, NUM_POD_MODES + 1),
                relative_importance_content[:NUM_POD_MODES],
                '-o',
                color='black',
                ms=8,
                alpha=1,
                mfc='red')
    axis[1].set_xlabel('k', fontdict=font)
    axis[1].set_ylabel(r'$RIC_{k}(\%)$', fontdict=font)
    axis[1].grid()
    plt.savefig('Relative Importance Content.png')
    plt.show()    

    for i in range(NUM_POD_MODES):
        plt.scatter(x_coord, y_coord, c=uPOD[i, :], cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.xlabel('x', fontdict=font)
        plt.ylabel('y', fontdict=font)
        plt.suptitle('u - POD Mode {}'.format(i+1), fontdict=font)
        # plt.savefig('u - POD Mode {}.png'.format(i+1))
        plt.show()

