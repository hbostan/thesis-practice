import numpy as np


class SnapshotData:

    def __init__(self, xs, ys, us, vs):
        self.xs = xs
        self.ys = ys
        self.us = us
        self.vs = vs
        self.U = np.vstack((us, vs))


class Snapshots:

    def __init__(self, m):
        self.time = []
        for mesh in m:
            xs = [n.x for n in mesh.nodes]
            ys = [n.y for n in mesh.nodes]
            us = [n.u_value for n in mesh.nodes]
            vs = [n.v_value for n in mesh.nodes]
            self.time.append(SnapshotData(xs, ys, us, vs))
        self.num_snaps = len(self.time)