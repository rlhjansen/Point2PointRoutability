import argparse
import os

from classes.mesh import Mesh, file_to_mesh


def create_net_datapath(c, n, x, y):
    abspath = create_circuit_datapath(c, x, y)
    abspath = os.path.join(abspath, "N"+str(n))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    abspath = os.path.join(abspath, "N"+str(n)+"_"+str(len(os.listdir(abspath)))+".csv")
    open(abspath, "a").close()
    return abspath


def create_circuit_datapath(c, x, y):
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
    abspath = os.path.join(abspath, "x"+str(x)+"y"+str(y))

    if not os.path.exists(abspath):
        os.makedirs(abspath)
    abspath = os.path.join(abspath, 'C'+str(c))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    print(abspath)
    return abspath


def main(x, y, netlens, netcount, terminalcount=100):
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
    abspath = os.path.join(abspath, "x"+str(x)+"y"+str(y))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    newmesh = Mesh([x, y])
    newmesh.generate_terminals(terminalcount)
    circuit_path = create_circuit_datapath(terminalcount, x, y) + ".csv"
    newmesh.write_mesh(circuit_path)
    for n in netlens:
        for _ in range(netcount):
            netlistpath = create_net_datapath(terminalcount, n, x, y)
            newmesh.generate_nets(n)
            newmesh.write_nets(netlistpath)
            newmesh.wipe_nets()

def make_config(pathlist_count, meshsizes, netlist_lengths):
    cfg_dict = {"pathlist_count":pathlist_count, "meshsizes":meshsizes, "netlist_lengths":netlist_lengths}
    with open("config.txt", "w+") as f:
        f.write(str(cfg_dict))

if __name__ == '__main__':
    netlist_lengths = [10+i for i in range(81)]
    meshsizes = [20,30,40,50,60,70,80,90,100]
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pathlist_count", help="the number of different pathlists to use", type=int, default=20)
    args = parser.parse_args()

    make_config(args.pathlist_count, meshsizes, netlist_lengths)

    for meshsize in meshsizes:
        main(meshsize, meshsize, netlist_lengths, args.pathlist_count)
