import os

from .mesh import file_to_mesh


class Gatherer:
    def __init__(self, c, n, nX, x, y, routings_per_pathlist):
        self.c = c      # mesh with c terminals
        self.n = n      # netlist with n nets
        self.nX = nX    # Xth netlist with n nets
        self.x = x
        self.y = y
        self.routings_per_pathlist = routings_per_pathlist
        self.all_data = []
        self.setup_load_paths()


    def add_iter_batch(self, batch_results):
        iter = str(self.iter)
        self.all_data.extend([[iter] + br for br in batch_results])


    def setup_load_paths(self):
        abspath = os.path.abspath(__file__)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.join(abspath, "data")
        abspath = os.path.join(abspath, "x"+str(self.x)+"y"+str(self.y))
        abspath = os.path.join(abspath, 'C'+str(self.c))
        self.mesh_path = abspath+".csv"
        abspath = os.path.join(abspath, "N"+str(self.n))
        abspath = os.path.join(abspath, "N"+str(self.n)+"_"+str(self.nX)+".csv")
        self.netlist_path = abspath


    def set_saveloc(self):
        abspath = os.path.abspath(__file__)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.join("results")
        abspath = os.path.join(abspath, "x"+str(self.x)+"y"+str(self.y))

        abspath = os.path.join(abspath, 'C'+str(self.c))
        if not os.path.exists(abspath):
            os.makedirs(abspath)
        abspath = os.path.join(abspath, "N"+str(self.n))
        abspath = os.path.join(abspath, "N"+str(self.n)+"_"+str(self.nX))
        if not os.path.exists(abspath):
            os.makedirs(abspath)

        prevdata = len(os.listdir(abspath))
        if not os.path.exists(abspath):
            os.makedirs(abspath)
        self.savedir = abspath



    def save_all_data(self):
        save_data = [";".join([d[0], str(d[1]), str(d[2]), ",".join(d[3:])]) for d in self.all_data]
        savefile = os.path.join(self.savedir, "all_data.csv")
        with open(savefile, "w+") as f:
            for line in save_data:
                f.write(line + "\n")
