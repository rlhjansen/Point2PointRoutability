import argparse

from statistics import mean
import os

from common import read_config


def reorder_by_netlength(files, lengths, iters, chipsize):
    """ Groups files by length and gathers individual filenames
    """
    netlendict = {k:{"filenames":[], 'count':0} for k in lengths}
    for f in files:

        filedata = f.split(os.sep)
        net_data = filedata[5].split('_')[0][1:]
        k = net_data
        netlendict[int(k)]["filenames"].append(f)
    return netlendict


def get_files(xsize, ysize, iters):
    path = os.path.curdir
    path = os.path.join(path, "results")
    path = os.path.join(path, "x" + str(xsize) + "y" + str(ysize))
    path = os.path.join(path, "C100")

    return [os.path.join(fdata[0], fdata[2][fdata[2].index('all_data.csv')]) for fdata in os.walk(path) if 'all_data.csv' in fdata[2]]


def get_first_placed_count(f):
    """ Gets the first scatterpoint from a file
    """
    readfile = open(f, 'r')
    line = readfile.readline()
    data = line.split(";")
    firstcount = int(data[1])
    readfile.close()
    return firstcount

def get_least_placed_count(f, iters):
    """ Gets the best scatterpoint from a file
    """
    readfile = open(f, 'r')
    best_count = None
    for i, line in enumerate(readfile.readlines()):
        data = line.split(";")
        count = int(data[1])
        if i == iters:
            break
        if i == 0:
            best_count = count
        else:
            if count < best_count:
                best_count = count
    readfile.close()
    return best_count


def get_most_placed_count(f, iters):
    """ Gets the best scatterpoint from a file
    """
    readfile = open(f, 'r')
    best_count = None
    for i, line in enumerate(readfile.readlines()):
        if i == iters:
            break
        data = line.split(";")
        count = int(data[1])
        if i == 0:
            best_count = count
        else:
            if count > best_count:
                best_count = count
    readfile.close()
    return best_count

def get_mean_routed_count(f, k, iters):
    """ Gets the scatterpoints from a file
    """
    readfile = open(f, 'r')
    place_counts = []
    for i, line in enumerate(readfile.readlines()):
        if i == iters:
            break
        data = line.split(";")
        place_counts.append(1 if int(data[1]) == k else 0)
    readfile.close()
    return mean(place_counts)



def make_netlen_routabilitydict(files, netlengths, netlendict, chipsize, iters):
    netlen_routabilitydict = {k:{'f':[], 'bc':[], 'mc':[], 'minc':[], 'count':0} for k in netlengths}
    for k in netlendict:

        for f in netlendict[k]["filenames"]:
            firstcount = get_first_placed_count(f)
            best_count = get_most_placed_count(f, iters)
            mean_count = get_mean_routed_count(f, k, iters)
            min_count = get_least_placed_count(f, iters)
            netlen_routabilitydict[k]['f'].append(1 if firstcount == k else 0)
            netlen_routabilitydict[k]['mc'].append(mean_count)
            netlen_routabilitydict[k]['bc'].append(1 if best_count == k else 0)
            netlen_routabilitydict[k]['minc'].append(1 if min_count == k else 0)
    return netlen_routabilitydict



def gather_data_chipsize(chipsize, netlengths, iters):
    files = get_files(chipsize, chipsize, iters)
    netlendict = reorder_by_netlength(files, netlengths, iters, chipsize)
    netlen_routability_dict = make_netlen_routabilitydict(files, netlengths, netlendict, chipsize, iters)
    return files, netlendict, netlen_routability_dict


def routability_header_gen(chipsizes, best_of_N):
    for cs in chipsizes:
        random_routability = ["routability by arb {}x{}".format(str(cs), str(cs))]
        mean_routability = ["routability of mean {}x{}".format(str(cs), str(cs))]
        best_routability = ["routability best of {}x{}".format(str(cs), str(cs))]
        worst_routability = ["routability worst of {}x{}".format(str(cs), str(cs))]
        yield random_routability
        yield mean_routability
        yield best_routability
        yield worst_routability


def make_routability_csvs(chipsizes, best_of_N, netlengths):
    netlendicts_persize = []
    netlen_countdicts_persize = []
    csv_data_walk = [["netlist length"]] + [elem for elem in routability_header_gen(chipsizes, best_of_N)]
    dw_len = len(csv_data_walk)
    csv_data_walk[0].extend([str(nl) for nl in netlengths])
    for i, chipsize in enumerate(chipsizes):
        j = i*4
        files, netlendict, netlen_routabilitydict = gather_data_chipsize(chipsize, netlengths, best_of_N)
        csv_data_walk[j+1].extend([str(mean(netlen_routabilitydict[n]['f'])) for n in netlengths])
        csv_data_walk[j+2].extend([str(mean(netlen_routabilitydict[n]['mc'])) for n in netlengths])
        csv_data_walk[j+3].extend([str(mean(netlen_routabilitydict[n]['bc'])) for n in netlengths])
        csv_data_walk[j+4].extend([str(mean(netlen_routabilitydict[n]['minc'])) for n in netlengths])
    with open("compare_routability_best_of_"+str(best_of_N)+".csv", "w+") as f:
        for i, netlength in enumerate(csv_data_walk[0]):
            line = ",".join([csv_data_walk[j][i] for j in range(dw_len)]) + "\n"
            f.write(line)


if __name__ == '__main__':
    config_dict = read_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--routings_per_pathlist", help="the number of different orders each pathlist should be routed, the paper uses 200", type=int, default=config_dict['routings_per_pathlist'])
    args = parser.parse_args()

    N = args.routings_per_pathlist
    meshsizes = config_dict['meshsizes']
    netlist_lengths = config_dict['netlist_lengths']

    make_routability_csvs(meshsizes, N, netlist_lengths)
