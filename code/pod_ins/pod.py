import numpy as np
from numpy import linalg as LA


class PODResult:

    def __init__(self, uMean, vMean, uPOD, vPOD, tc, npm):
        self.uMean = uMean
        self.vMean = vMean
        self.uPOD = uPOD
        self.vPOD = vPOD
        self.time_coeff = tc
        self.num_pod_modes = npm


def calculate_pod_modes(ts_dgnu, ts_dgnv, ts_dgnp, num_ts, num_nodes, num_pod_modes):
    U = np.zeros((num_ts, num_nodes))
    V = np.zeros((num_ts, num_nodes))
    P = np.zeros((num_ts, num_nodes))

    for i in range(num_ts):
        U[i, :] = ts_dgnu[i].flatten()
        V[i, :] = ts_dgnv[i].flatten()
        P[i, :] = ts_dgnp[i].flatten()

    # Find mean velocities wrt time
    uMean = np.mean(U, axis=0)
    vMean = np.mean(V, axis=0)
    pMean = np.mean(P, axis=0)

    # Substract mean from real velocities to find fluctuating velocities.
    uFluc = np.zeros((num_ts, num_nodes))
    vFluc = np.zeros((num_ts, num_nodes))
    for t in range(num_ts):
        uFluc[t, :] = U[t, :] - uMean
        vFluc[t, :] = V[t, :] - vMean

    # Create snapshot array using fluctuating velocities.
    snapshot = np.concatenate((uFluc, vFluc), axis=1)
    covariance = np.cov(snapshot)

    # Eigenvalues and eigenvector of correlation matrix.
    eig_vals, eig_vecs = LA.eig(covariance)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # Sort eigenvalues and vectors according to abs(eig_val)
    sorting_indices = np.abs(eig_vals).argsort()[::-1]
    eig_vals = eig_vals[sorting_indices]
    eig_vecs = eig_vecs[:, sorting_indices]

    # Find and normalize time coefficients
    time_coeff = np.zeros((num_ts, num_pod_modes))
    for i in range(num_pod_modes):
        for j in range(num_ts):
            time_factor = np.sqrt(num_ts * eig_vals[i])
            time_coeff[j][i] = eig_vecs[j][i] * time_factor

    # Calculate POD modes
    uPOD = np.zeros((num_pod_modes, num_nodes))
    vPOD = np.zeros((num_pod_modes, num_nodes))
    for i in range(num_pod_modes):
        for j in range(num_ts):
            uPOD[i, :] = uPOD[i, :] + eig_vecs[j][i] * uFluc[j, :]
            vPOD[i, :] = vPOD[i, :] + eig_vecs[j][i] * vFluc[j, :]
        mode_factor = 1 / np.sqrt(num_ts * eig_vals[i])
        uPOD[i, :] = uPOD[i, :] * mode_factor
        vPOD[i, :] = vPOD[i, :] * mode_factor

    return PODResult(uMean, vMean, uPOD, vPOD, time_coeff, num_pod_modes)
