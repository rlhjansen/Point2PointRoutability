
import multiprocessing as mp
import sys
import argparse
import datetime

from common import read_config

import classes.simpleRouter as r


def router_generator(meshsize, netlist_length, pathlist_count, routings_per_pathlist, num_terminals=100):
    print("creating router objects for meshsize", meshsize, "netlist length", netlist_length)
    for nX in range(pathlist_count):

        s = r.Router(num_terminals, netlist_length, nX, meshsize,meshsize, routings_per_pathlist)
        yield s

def start_wrapper(router_obj):
    router_obj.route()



def append_routings_per_pathlist_to_config(config, routings_per_pathlist):
    config["routings_per_pathlist"] = routings_per_pathlist
    with open("config.txt", "w+") as f:
        f.write(str(config))

if __name__ == '__main__':
    config_dict = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--routings_per_pathlist", help="the number of different orders each pathlist should be routed, the paper uses 200", type=int, default=200)
    parser.add_argument("-c", "--cpu_count", help="number of cpus to use, default is computer cpus -2, so other things can still run (slowly) besides the experiment", type=int, default=mp.cpu_count()-2)
    args = parser.parse_args()

    append_routings_per_pathlist_to_config(config_dict, args.routings_per_pathlist)

    starttime = datetime.datetime.now()
    for meshsize in config_dict["meshsizes"]:
        for netlist_length in config_dict["netlist_lengths"]:

            pool = mp.Pool(args.cpu_count)
            Routers = router_generator(meshsize, netlist_length, config_dict["pathlist_count"], args.routings_per_pathlist)
            pool.map(start_wrapper, Routers)
            pool.close()

    print("time elapsed:\t", datetime.datetime.now() - starttime)
