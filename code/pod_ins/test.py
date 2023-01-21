import meshio
import matplotlib.pyplot as plt
from geometry.mesh.cartesian_mesh import CartesianStructuredMesh
import utils.cylinder_plot as cp

meshio_mesh = meshio.read('test.vtu')
cartesian_mesh = CartesianStructuredMesh(meshio_mesh)

xs = [n.x for n in cartesian_mesh.nodes]
ys = [n.y for n in cartesian_mesh.nodes]
val = [n.u_value for n in cartesian_mesh.nodes]
dx, dy = cartesian_mesh.finite_differences(data=val)
# plt.scatter(xs, ys, c=val)
cp.plot_cylinder_data(xs, ys, dy)
plt.show()
