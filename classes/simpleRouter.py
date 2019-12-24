
from .mesh import file_to_mesh
from .gatherer import Gatherer

import matplotlib.pyplot as plt


class Router(Gatherer):
    def __init__(self, c, n, nX, x, y, routings_per_pathlist):

        Gatherer.__init__(self, c, n, nX, x, y, routings_per_pathlist)
        self.set_saveloc()
        self.mesh = file_to_mesh(self.mesh_path, None)
        self.mesh.read_nets(self.netlist_path)



    def route(self):

        self.mesh.connect()
        self.iter = 0
        ords = [self.mesh.get_random_net_order() for i in range(self.routings_per_pathlist)]
        data = [self.mesh.solve_order(ords[i], reset=True)[:2] for i in range(self.routings_per_pathlist)]
        combined_data = [data[i] + ords[i] for i in range(len(data))]

        self.add_iter_batch(combined_data)
        self.save_all_data()
        print("running object with meshsize", self.x, "netlistlen", self.n, "pathlist", self.nX)

        return
