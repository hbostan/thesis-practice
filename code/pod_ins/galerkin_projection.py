import numpy as np
from pod import PODResult
from topology.mesh import Mesh
from scipy import integrate


def pod_mode_derivatives(mesh, uPOD, vPOD, num_pod_modes, num_elements):
    uPOD_1st_dx = np.zeros((num_pod_modes, num_elements))
    uPOD_1st_dy = np.zeros((num_pod_modes, num_elements))
    vPOD_1st_dx = np.zeros((num_pod_modes, num_elements))
    vPOD_1st_dy = np.zeros((num_pod_modes, num_elements))

    uPOD_2nd_dx = np.zeros((num_pod_modes, num_elements))
    uPOD_2nd_dy = np.zeros((num_pod_modes, num_elements))
    vPOD_2nd_dx = np.zeros((num_pod_modes, num_elements))
    vPOD_2nd_dy = np.zeros((num_pod_modes, num_elements))
    # POD MODE DERIVATIVES
    for i in range(num_pod_modes):
        u = uPOD[i, :, :]
        v = vPOD[i, :, :]

        u_1st_dx, u_1st_dy, v_1st_dx, v_1st_dy = mesh.first_derivative(u, v)
        u_2nd_dx, u_2nd_dy, v_2nd_dx, v_2nd_dy = mesh.second_derivative(u, v)

        uPOD_1st_dx[i, :] = u_1st_dx
        uPOD_1st_dy[i, :] = u_1st_dy
        vPOD_1st_dx[i, :] = v_1st_dx
        vPOD_1st_dy[i, :] = v_1st_dy

        uPOD_2nd_dx[i, :] = u_2nd_dx
        uPOD_2nd_dy[i, :] = u_2nd_dy
        vPOD_2nd_dx[i, :] = v_2nd_dx
        vPOD_2nd_dy[i, :] = v_2nd_dy
    return ((uPOD_1st_dx, uPOD_1st_dy, vPOD_1st_dx, vPOD_1st_dy), (uPOD_2nd_dx, uPOD_2nd_dy, vPOD_2nd_dx, vPOD_2nd_dy))


def galerkin_projection(mesh, pod_result, viscosity, t_final, timestep):
    uMean, vMean = pod_result.uMean, pod_result.vMean
    uPOD, vPOD = pod_result.uPOD, pod_result.vPOD
    num_pod_modes = pod_result.num_pod_modes

    num_elements = mesh.num_triangles
    num_node_per_element = mesh.num_node_per_triangle

    uPOD = uPOD.reshape(num_pod_modes, num_elements, num_node_per_element)
    vPOD = vPOD.reshape(num_pod_modes, num_elements, num_node_per_element)
    uMean = uMean.reshape(num_elements, num_node_per_element)
    vMean = vMean.reshape(num_elements, num_node_per_element)

    uMean_1st_dx, uMean_1st_dy, vMean_1st_dx, vMean_1st_dy = mesh.first_derivative(uMean, vMean)
    uMean_2nd_dx, uMean_2nd_dy, vMean_2nd_dx, vMean_2nd_dy = mesh.second_derivative(uMean, vMean)
    pod_mode_1st_derivatives, pod_mode_2nd_derivatives = pod_mode_derivatives(mesh, uPOD, vPOD, num_pod_modes,
                                                                              num_elements)

    uPOD_1st_dx, uPOD_1st_dy, vPOD_1st_dx, vPOD_1st_dy = pod_mode_1st_derivatives
    uPOD_2nd_dx, uPOD_2nd_dy, vPOD_2nd_dx, vPOD_2nd_dy = pod_mode_2nd_derivatives

    # TODO move these to centers
    uMean = mesh.get_value_at_centers(uMean)
    vMean = mesh.get_value_at_centers(vMean)
    # TODO move these to centers
    uPOD_centers = np.zeros((num_pod_modes, num_elements))
    vPOD_centers = np.zeros((num_pod_modes, num_elements))
    for i in range(num_pod_modes):
        uPOD_centers[i] = mesh.get_value_at_centers(uPOD[i])
        vPOD_centers[i] = mesh.get_value_at_centers(vPOD[i])
    uPOD = uPOD_centers
    vPOD = vPOD_centers

    areas = mesh.get_areas()

    constant = np.zeros(num_pod_modes)
    for k in range(num_pod_modes):
        constant[k] = np.sum(np.sum(areas*(viscosity * (uMean_2nd_dx + uMean_2nd_dy) * uPOD[k, :] \
                                + viscosity * (vMean_2nd_dx + vMean_2nd_dy) * vPOD[k, :] \
                                - (uMean * uMean_1st_dx + vMean * uMean_1st_dy) * uPOD[k, :] \
                                - (uMean * vMean_1st_dx + vMean * vMean_1st_dy) * vPOD[k, :])))

    linear = np.zeros((num_pod_modes, num_pod_modes))
    for k in range(num_pod_modes):
        for m in range(num_pod_modes):
            linear[m, k] = np.sum(np.sum(areas*(-(uMean * uPOD_1st_dx[m, :] + vMean * uPOD_1st_dy[m, :]) * uPOD[k, :] \
                                -(uMean * vPOD_1st_dx[m, :] + vMean * vPOD_1st_dy[m, :]) * vPOD[k, :] \
                                -(uPOD[m, :] * uMean_1st_dx + vPOD[m, :] * uMean_1st_dy) * uPOD[k, :] \
                                -(uPOD[m, :] * vMean_1st_dx + vPOD[m, :] * vMean_1st_dy) * vPOD[k, :] \
                                + viscosity * (uPOD_2nd_dx[m, :] + uPOD_2nd_dy[m, :]) * uPOD[k, :] \
                                + viscosity * (vPOD_2nd_dx[m, :] + vPOD_2nd_dy[m, :]) * vPOD[k, :])))

    non_linear = np.zeros((num_pod_modes, num_pod_modes, num_pod_modes))
    for k in range(num_pod_modes):
        for m in range(num_pod_modes):
            for n in range(num_pod_modes):
                non_linear[n, m, k] = -np.sum(np.sum(areas*((uPOD[m, :] * uPOD_1st_dx[n, :] + vPOD[m, :] * uPOD_1st_dy[n, :]) * uPOD[k, :] \
                                    + (uPOD[m, :] * vPOD_1st_dx[n, :] + vPOD[m, :] * vPOD_1st_dy[n, :]) * vPOD[k, :])))

    def rhs(tINT, TimeCoeffGalerkin):

        sum_linear = np.zeros(num_pod_modes, dtype=np.float64)
        sum_non_linear = np.zeros(num_pod_modes, dtype=np.float64)

        for k in range(num_pod_modes):
            for m in range(num_pod_modes):
                sum_linear[k] += linear[m, k] * TimeCoeffGalerkin[m]

        for k in range(num_pod_modes):
            for m in range(num_pod_modes):
                for n in range(num_pod_modes):
                    sum_non_linear[k] += non_linear[n, m, k] * TimeCoeffGalerkin[n] * TimeCoeffGalerkin[m]

        return constant + sum_linear + sum_non_linear

    t0, t1 = 0, t_final  # start and end
    tInt = np.arange(t0, t1 + timestep, timestep)  # the points of evaluation of solution
    TimeCoeffIC = pod_result.time_coeff[0, :]  # initial value
    TimeCoeffGalerkin = np.zeros((len(tInt), len(TimeCoeffIC)))  # array for solution

    TimeCoeffGalerkin[0, :] = TimeCoeffIC

    r = integrate.ode(rhs).set_integrator("dop853", nsteps=500)  # choice of method
    r.set_initial_value(TimeCoeffIC, t0)  # initial values
    for i in range(1, tInt.size):
        print(f'Time: {i}', end='\r')
        TimeCoeffGalerkin[i, :] = r.integrate(tInt[i])  # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")
    print()
    return TimeCoeffGalerkin, tInt