import argparse

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares
from math import exp
import numpy as np

from common import read_config

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

BBOX_TO_ANCHOR = (0.5, 0.45, .4, .55)

LEGEND_LOC = 4 #location given the bounding box
LEGEND_WINDOW_INDEX = 1 #location of legend in the 3x3 grid figure, as follows

# 0 # 1 # 2
# 3 # 4 # 5
# 6 # 7 # 8

initial_col = '#1b9e77'
fit_col = '#1b9e77'
best_fit = '#d95f02'
best_col = '#d95f02'
mean_col = 'b'
worst_col = 'magenta'


def mean(list):
    return sum(list)/len(list)


def named_function(name):
    def naming(func):
        func.name = name
        return func
    return naming


def format_meshsize(meshsize):
    return str(meshsize)+"x"+str(meshsize)


def initial_solv_str(meshsize):
    return "routability by arb {}x{}".format(str(meshsize), str(meshsize))


def best_solv_str(meshsize):
    return "routability best of {}x{}".format(str(meshsize), str(meshsize))


def worst_solv_str(meshsize):
    return "routability worst of {}x{}".format(str(meshsize), str(meshsize))


def mean_solv_str(meshsize):
    return "routability of mean {}x{}".format(meshsize, meshsize)


@named_function("balanced sigmoid")
def inverted_logistic(nl, shift, slope):
    return 1 - (1 /(1+ np.exp(-(nl-shift)*slope)))


def lstr(iterable):
    return [str(elem) for elem in iterable]


# 3x3 window location checker, used to assign labels/axis values
def determine_3x3_y(elem_n):
    return not elem_n % 3


def determine_3x3_solv(elem_n):
    return elem_n == 3


def determine_3x3_nl(elem_n):
    return elem_n == 7


def determine_3x3_x(elem_n):
    return elem_n // 6


def save_shift_slope(meshsizes, param_func, N):
    fitfunc = eval(param_func)
    datafile = "compare_routability_best_of_"+str(N)+".csv"
    fit_error_file = "routability_fits_per_meshsize.csv"
    fit_error_file_lines = ["meshsize, initial routability param, initial slope param, initial squared error avg, after routability param, after slope param, after squared error avg"]
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    param_csv = open("params_" + param_func + ".csv", "w+")
    param_csv.write(", arbitrary netlists,,mean routability,,optimized netlist order,,worst order,\n")
    param_csv.write("meshsize (XxZ),initial_a,initial_b,mean_a,mean_b,best_a,best_b,worst_a,worst_b\n")
    errorfunc = lambda vars,x,data : fitfunc(x, vars[0], vars[1]) - data
    for j, meshsize in enumerate(meshsizes):
        y_arb = df[initial_solv_str(meshsize)]
        y_mean = df[mean_solv_str(meshsize)]
        y_best = df[best_solv_str(meshsize)]
        y_worst = df[worst_solv_str(meshsize)]

        fitResArb = least_squares(errorfunc, x0=(20, 0.05), args=(nl, y_arb), method='lm')
        fitResBest = least_squares(errorfunc, x0=(30, 0.05), args=(nl, y_best), method='lm')
        fitResMean = least_squares(errorfunc, x0=(20, 0.05), args=(nl, y_mean), method='lm')
        fitResWorst = least_squares(errorfunc, x0=(20, 0.05), args=(nl, y_worst), method='lm')
        fit_error_file_lines.append(",".join([str(meshsize), str(fitResArb.x[0]), str(fitResArb.x[1]), str(np.mean(np.power(fitResArb.fun, 2))), str(fitResBest.x[0]), str(fitResBest.x[1]), str(np.mean(np.power(fitResBest.fun, 2)))]))

        param_csv.write(",".join([str(meshsize)]+lstr(list(fitResArb.x))+lstr(list(fitResMean.x))+lstr(list(fitResBest.x))+lstr(list(fitResWorst.x)))+"\n")
    param_csv.close()
    with open(fit_error_file, "w+") as f:
        f.write("\n".join(fit_error_file_lines))

def conditional_label(boolean_value, label):
    if boolean_value:
        return label
    else:
        return None

def gen_filename_window(param_func, types, scatter, fitted):
    if not os.path.exists(param_func):
        os.mkdir(param_func)
    plot_savefile = param_func + "/" + "_".join(types)
    if scatter:
        plot_savefile += "_s"
    if fitted:
        plot_savefile += "_f"
    plot_savefile += "_3x3.png"
    return plot_savefile

def plot_shift_slope(meshsizes, types, param_func, title, scatter=True, fitted=True, legend=True):
    fitfunc = eval(param_func)
    plot_savefile = gen_filename_window(param_func, types, scatter, fitted)

    datafile = "compare_routability_best_of_"+str(N)+".csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())
    ab_df = load_shift_slope(param_func)

    params_r = []
    params_b = []
    params_m = []
    params_w = []

    _best = "best" in types
    _mean = "mean" in types
    _arb = "initial" in types
    _worst = "worst" in types
    fig=plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(3,3)
    gs.update(wspace=0.02, hspace=0.02) # set the spacing between axes.

    legend_loc = 0

    for j, cs in enumerate(meshsizes):
        ax = plt.subplot(gs[j])
        # ax = plt.subplot2mesh((3,3), (j//3, j%3))
        ax.text(.95,.9,format_meshsize(cs),horizontalalignment='right', transform=ax.transAxes)
        if not determine_3x3_x(j):
            ax.set_xticks([])
        else:
            ax.set_xticks([10,20,30,40,50,60,70,80,90])
        if determine_3x3_nl(j):
            ax.set_xlabel("Number of paths in pathlist")
        if not determine_3x3_y(j):
            ax.set_yticks([])
        if determine_3x3_solv(j):
            ax.set_ylabel("Routability")

        labelwindow = j==legend_loc

        if _arb:
            y_arb = df[initial_solv_str(cs)]
            popta = ab_df["initial_a"][j], ab_df["initial_b"][j]
            if scatter:
                plt.scatter(nl, y_arb, c=mean_col, s=6, label=conditional_label(labelwindow, "original"))
            if fitted:
                ABNL_plot(nl, popta, fitfunc, c='k')

        if _mean:
            y_mean = df[mean_solv_str(cs)]
            poptm = ab_df["mean_a"][j], ab_df["mean_b"][j]
            if scatter:
                plt.scatter(nl, y_mean, c=mean_col, s=6, label=conditional_label(labelwindow, "average sequence routability"))
            if fitted:
                ABNL_plot(nl, poptm, fitfunc, c=fit_col)

        if _best:
            y_best = df[best_solv_str(cs)]
            poptb = ab_df["best_a"][j], ab_df["best_b"][j]
            if scatter:
                plt.scatter(nl, y_best, c=best_col, s=6, label=conditional_label(labelwindow, "permuted"))
            if fitted:
                ABNL_plot(nl, poptb, fitfunc, c='k')

        if _worst:
            y_worst = df[best_solv_str(cs)]
            poptw = ab_df["worst_a"][j], ab_df["worst_b"][j]
            if scatter:
                plt.scatter(nl, y_worst, c=best_col, s=6, label=conditional_label(labelwindow, "worst after permutation"))
            if fitted:
                ABNL_plot(nl, poptw, fitfunc, c='k')


        if labelwindow and legend:
            # Put a legend to the right of the current axis

            lgd = plt.legend(bbox_to_anchor=BBOX_TO_ANCHOR, loc=LEGEND_LOC, fancybox=False, shadow=False, ncol=1, frameon=False)


            # original settings
            # also set legend_loc to 7 above
            # lgd = plt.legend(bbox_to_anchor=(1, 1.0))
            # plt.legend(loc='upper center',
            #  bbox_to_anchor=(0.5, -0.22),fancybox=False, shadow=False, ncol=3)
    plt.suptitle(title)
    if legend:
        # tight bounding box
        plt.savefig(plot_savefile, bbox_extra_artists=(lgd,))
    else:
        plt.savefig(plot_savefile)
    plt.show()


def ABNL_plot(nl, popts, fitfunc, c, label=None):
    if label:
        plt.plot(nl, fitfunc(nl, *popts), c=c, linestyle='--', label=label)
    else:
        plt.plot(nl, fitfunc(nl, *popts), c=c, linestyle='--')


def load_shift_slope(param_func):
    df = pd.read_csv("params_" + param_func + ".csv", header=1)
    return df



def getdfcol(df, n):
    """ get n'th column values of dataframe
    """
    return df[df.columns[n]]



if __name__ == '__main__':
    config_dict = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--routings_per_pathlist", help="the number of different orders each pathlist should be routed, the paper uses 200", type=int, default=config_dict['routings_per_pathlist'])
    args = parser.parse_args()
    N = args.routings_per_pathlist

    meshsizes = config_dict["meshsizes"]
    save_shift_slope(meshsizes, "inverted_logistic", N)

    plot_shift_slope(meshsizes, ["initial", "best"], "inverted_logistic", "", scatter=True)
