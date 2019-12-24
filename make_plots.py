
import sys
import argparse
from subprocess import run
from common import read_config

sys.path.append("..")

if __name__ == '__main__':
    config_dict = read_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--routings_per_pathlist", help="the number of different orders each pathlist should be routed, the paper uses 200", type=str, default=str(config_dict["routings_per_pathlist"]))
    args = parser.parse_args()
    N = args.routings_per_pathlist


    run(("python calculate_routability.py --routings_per_pathlist " + N).split(" "))
    run(("python fit_meshwise.py --routings_per_pathlist " + N).split(" "))
    run(("python fit_routability_parameters.py --routings_per_pathlist " + N).split(" "))
