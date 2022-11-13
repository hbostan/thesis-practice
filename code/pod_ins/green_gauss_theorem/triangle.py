from vec2 import Vec2
import numpy as np

class Triangle:

    def __init__(self, vertex_x, vertex_y, node_x, node_y):
        self.vertices: list[Vec2] = [
            Vec2(vertex_x[0], vertex_y[0]),
            Vec2(vertex_x[1], vertex_y[1]),
            Vec2(vertex_x[2], vertex_y[2]),
        ]
        self.edges: list[Vec2] = [
            (self.vertices[1] - self.vertices[0]),  # 0 - 02 -1%3
            (self.vertices[2] - self.vertices[1]),  # 1 - 10 0%3
            (self.vertices[0] - self.vertices[2]),  # 2 - 21 1%3
        ]
        self.edge_centers: list[Vec2] = [
            (self.vertices[1] + self.vertices[0]) / 2,
            (self.vertices[2] + self.vertices[1]) / 2,
            (self.vertices[0] + self.vertices[2]) / 2,
        ]
        self.neighbors: list['Triangle'] = []
        self.interp_coeff: list[float] = []

        self.nodes: list[Vec2] = [
            Vec2(nx, ny) for nx, ny in zip(node_x, node_y)
        ]
        self.center: Vec2 = self.find_center()
        self.area : float = self.find_area()
        self.normals: list[Vec2] = self.find_edge_normals()

    def get_vertex_coords(self):
        return np.array([[v.x, v.y] for v in self.vertices])

    def add_neighbor(self, neighbor: 'Triangle'):
        self.neighbors.append(neighbor)
        if neighbor == None:
            self.interp_coeff.append(0)
            return
        # coeff = |(xf -xn)| / |(xp - xn)|
        which_edge = len(self.neighbors) - 1 # self.neighbors.index(neighbor)
        center2edge = neighbor.center.distance_to(self.edge_centers[which_edge])
        center2center = self.center.distance_to(neighbor.center)
        coeff = center2edge/center2center
        self.interp_coeff.append(coeff)
        
    # CoM of the triangle as the average of x and y values
    def find_center(self) -> Vec2:
        x, y = 0, 0
        for v in self.vertices:
            x += v.x
            y += v.y
        x, y = x / 3, y / 3
        return Vec2(x, y)

    # Calculate the normals for each edge
    def find_edge_normals(self) -> list[Vec2]:
        normals = []
        for i, edge in enumerate(self.edges):
            other_point = self.vertices[(2 + i) % 3]
            other_line = other_point - self.vertices[i]
            normal = edge.normal()
            if other_line.dot(normal) > 0:
                normal = -1 * normal
            normal = normal.normalize()
            normals.append(normal)
        return normals

    # Takes the weighted average of values at each node in a cell
    # based on their distance to the cell center.
    def value_at_center(self, var) -> float:
        total_w = 0
        total_u = 0
        for i, v in enumerate(var):
            # Node weights and total weight can be precomputed
            # to speed up stuff
            p = self.nodes[i]
            w = 1 / p.distance_to(self.center)
            total_w += w
            total_u += w * v
        return total_u / total_w
    
    def find_area(self):
        v1, v2, v3 = self.vertices
        # Area by matrix determinant
        return (v1.x*(v2.y-v3.y)) + (v2.x*(v3.y-v1.y)) + (v3.x*(v1.y-v2.y))

    # Find and set the values of u, v, p at triangle (cell) center.
    def update(self, u, v, p):
        self.u_at_center = self.value_at_center(u)
        self.v_at_center = self.value_at_center(v)
        self.p_at_center = self.value_at_center(p)

    def u_find_first_derivative(self):
        first_derivative = Vec2(0, 0)
        for edge_idx, edge in enumerate(self.edges):
            coeff = self.interp_coeff[edge_idx]
            neighbor = self.neighbors[edge_idx]
            normal = self.normals[edge_idx]
            # if at boundary skip edge
            # -> inflow u_at_edge = Dirichlet BC (i.e. u_at_edge = 1)
            # -> outflow u_at_edge = Neumann BC (i.e. u_at_edge = u_at_center)
            # -> wall u_at_edge = Dirichlet BC (i.e. u_at_edge = 0)
            if neighbor == None:
                continue
            u_at_edge = (coeff * self.u_at_center) + ((1-coeff) * neighbor.u_at_center)
            flux_at_edge = u_at_edge * normal * edge.length
            first_derivative += flux_at_edge

        first_derivative = first_derivative / self.area
        self.u_first_derivative = first_derivative
        return first_derivative

    def u_find_second_derivative(self):
        second_derivative = Vec2(0, 0)
        # First derivative
        for n in self.neighbors:
            if n == None:
                continue
            n.u_find_first_derivative()
        self.u_find_first_derivative()
        # Second derivative
        for edge_idx, edge in enumerate(self.edges):
            neighbor = self.neighbors[edge_idx]
            if neighbor == None:
                continue
            normal = self.normals[edge_idx]
            coeff = self.interp_coeff[edge_idx]
            grad_at_edge = (coeff * self.u_first_derivative) + ((1-coeff) * neighbor.u_first_derivative)
            second_derivative += grad_at_edge * normal * edge.length
        second_derivative = second_derivative/ self.area
        self.u_second_derivative = second_derivative
        return second_derivative

    def v_find_first_derivative(self):
        first_derivative = Vec2(0, 0)
        for edge_idx, edge in enumerate(self.edges):
            coeff = self.interp_coeff[edge_idx]
            neighbor = self.neighbors[edge_idx]
            normal = self.normals[edge_idx]
            # if at boundary skip edge
            if neighbor == None:
                continue
            v_at_edge = (coeff * self.v_at_center) + ((1-coeff) * neighbor.v_at_center)
            flux_at_edge = v_at_edge * normal * edge.length
            first_derivative += flux_at_edge
        first_derivative = first_derivative / self.area
        self.v_first_derivative = first_derivative
        return first_derivative

    def v_find_second_derivative(self):
        second_derivative = Vec2(0, 0)
        # First derivative
        for n in self.neighbors:
            if n == None:
                continue
            n.v_find_first_derivative()
        self.v_find_first_derivative()
        # Second derivative
        for edge_idx, edge in enumerate(self.edges):
            neighbor = self.neighbors[edge_idx]
            if neighbor == None:
                continue
            normal = self.normals[edge_idx]
            coeff = self.interp_coeff[edge_idx]
            grad_at_edge = (coeff * self.v_first_derivative) + ((1-coeff) * neighbor.v_first_derivative)
            second_derivative += grad_at_edge * normal * edge.length
        second_derivative = second_derivative/ self.area
        self.v_second_derivative = second_derivative
        return second_derivative
    
    def p_find_first_derivative(self):
        grad = Vec2(0, 0)
        for edge_idx, edge in enumerate(self.edges):
            coeff = self.interp_coeff[edge_idx]
            neighbor = self.neighbors[edge_idx]
            normal = self.normals[edge_idx]
            # if at boundary skip edge
            if neighbor == None:
                continue
            p_at_edge = (coeff * self.p_at_center) + ((1-coeff) * neighbor.p_at_center)
            flux_at_edge = p_at_edge * normal * edge.length
            grad += flux_at_edge
        grad = grad / self.area
        return grad

class Inflow:
    def __init__(self, center, neighbor_of, u, v, p):
        self.center = center
        self.neighbor_of = neighbor_of
        self.u_at_center = u
        self.v_at_center = v 
        self.p_at_center = p

    # Upwind scheme to find the derivative at inlet
    def u_find_first_derivative(self):
        dudx = (self.u_at_center - self.neighbor_of.u_at_center)/ (self.center.distance_to(self.neighbor_of.center))
        dudy = 0
        self.u_first_derivative = Vec2(dudx, dudy)
        return self.u_first_derivative
    
    def v_find_first_derivative(self):
        dvdx = (self.v_at_center - self.neighbor_of.v_at_center)/ (self.center.distance_to(self.neighbor_of.center))
        dvdy = 0
        self.v_first_derivative = Vec2(dvdx, dvdy)
        return self.v_first_derivative

class Outflow:
    def __init__(self, center, neighbor_of):
        self.neighbor_of = neighbor_of
        self.center = center
    
    # Fully developed BC at outlet
    def u_find_first_derivative(self):
        self.u_first_derivative = Vec2(0,0)
        return self.u_first_derivative
    
    # Fully developed BC at outlet
    def v_find_first_derivative(self):
        self.v_first_derivative = Vec2(0,0)
        return self.v_first_derivative
    
    @property
    def u_at_center(self):
        return self.neighbor_of.u_at_center

    @property
    def v_at_center(self):
        return self.neighbor_of.v_at_center

    @property
    def p_at_center(self):
        return self.neighbor_of.p_at_center