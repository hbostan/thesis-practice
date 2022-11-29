from green_gauss_theorem.vec2 import Vec2
from green_gauss_theorem.triangle import Triangle, Inflow, Outflow
import numpy as np

U_INFLOW = 1
V_INFLOW = 0
P_INFLOW = 0


class Mesh:

    def __init__(self, mesh_info):
        self.points: list[Vec2] = self._find_unique_points(mesh_info)
        self.num_points = len(self.points)
        self.triangles: list[Triangle] = self._create_triangles(mesh_info)
        self._add_neighbor_info(mesh_info)

    def _find_unique_points(self, mesh_info):
        etov = np.array(mesh_info.EToV).flatten()
        num_pts = np.unique(etov).shape[0]
        points = [None] * num_pts
        for t_idx, verts in enumerate(mesh_info.EToV):
            x_coords = mesh_info.mesh_Vx[t_idx]
            y_coords = mesh_info.mesh_Vy[t_idx]
            for v_idx, x, y in zip(verts, x_coords, y_coords):
                if points[v_idx] == None:
                    points[v_idx] = Vec2(x, y)
        return points

    def _create_triangles(self, mesh_info):
        triangles = []
        for t_idx, pts in enumerate(mesh_info.EToV):
            t = Triangle(pts, self.points, mesh_info.mesh_DGNx[t_idx], mesh_info.mesh_DGNy[t_idx])
            triangles.append(t)
        return triangles

    def _add_neighbor_info(self, mesh_info):
        for t_idx, n_list in enumerate(mesh_info.EToE):
            for n_idx, n in enumerate(n_list):
                if n != -1:
                    neighbor = self.triangles[n]
                    self.triangles[t_idx].add_neighbor(neighbor)
                else:
                    boundary_type = mesh_info.EToB[t_idx][n_idx]
                    if boundary_type == 1:
                        self.triangles[t_idx].add_neighbor(None)
                    elif boundary_type == 2:
                        center = self.triangles[t_idx].edge_centers[n_idx]
                        inflow = Inflow(center, self.triangles[t_idx], U_INFLOW, V_INFLOW, P_INFLOW)
                        self.triangles[t_idx].add_neighbor(inflow)
                    elif boundary_type == 3:
                        center = self.triangles[t_idx].edge_centers[n_idx]
                        outflow = Outflow(center, self.triangles[t_idx])
                        self.triangles[t_idx].add_neighbor(outflow)
                    else:
                        self.triangles[t_idx].add_neighbor(None)

    def bbox(self):
        px = [v.x for v in self.points]
        py = [v.y for v in self.points]
        minx = min(px)
        maxx = max(px)
        miny = min(py)
        maxy = max(py)
        return ((minx, maxx), (miny, maxy))

    def update(self, u, v, p):
        for t_idx, t in enumerate(self.triangles):
            t.update(u[t_idx], v[t_idx], p[t_idx])