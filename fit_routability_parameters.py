import os
import argparse

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import exp, floor
import numpy as np

from common import read_config


def named_function(name):
    def naming(func):
        func.name = name
        return func
    return naming


def load_ab(param_func):
    df = pd.read_csv("params_" + param_func + ".csv", header=1)
    return df

def getdfcol(df, n):
    """ get n'th column values of dataframe
    """
    return df[df.columns[n]]


INTERP_MESH = np.array([i for i in range(400, 10001, 1)])

initial_col = 'b'
best_col = '#d95f02'
mean_col = 'g'
worst_col = 'magenta'
black = 'k'

meta_fit_vars = {}

meta_fit_vars['start'] = (13, 0.005, 0.1)
meta_fit_vars['start_s'] = (-0.8, 0.0005, 0.0001)
MFV = meta_fit_vars

@named_function('arg1 * np.log( arg2 * area - arg3)')
def logfunc(value, const1, const2, const3):
    return  const1 * np.log( const2 *value - const3)

@named_function('arg1 * np.log( arg2 * area - arg3)')
def slope_adapted_logfunc(value, const1, const2, const3):
    """shenanigans to make sure optimizer stays in the > 0 domain for the log pls yannick?"""
    if np.sum((const2 *value - const3) < 0):
        const4 = const3*-1
        return const1 * np.log( const2 *value - const4)

    return  const1 * np.log( const2 *value - const3)

def correct_slopefunc(value, const1, const2, const3):
    """ extensie van shenanigans om te checken voor correctness """
    if np.sum((const2 *value - const3) < 0):
        const4 = const3*-1
        return const1, const2, const4
    else:
        return const1, const2, const3



def log10(num):
    return np.log(num)/np.log(10)

def get_significant(x, n):
   r = round(x, -int(floor(log10(abs(x)))) + (n))
   return r


def format_found_params_inside_parenthesis(const1, const2, const3):
    return "r'" + str(get_significant(const1, 2)) + " $\cdot$ ln (" + str(get_significant(const2, 2)) + "x - " + str(get_significant(const3, 2)) + ")'"


def meta_fit(mesh_metric, param_func, meta_param_func, _arb=False, _mean=False, _best=False, _worst=False):

    meta_param_func_str = meta_param_func
    meta_param_func = eval(meta_param_func)

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    df = load_ab("inverted_logistic")
    mesh_area_values = getdfcol(df,0)**2
    initial_shift = getdfcol(df,1)
    best_shift = getdfcol(df,5)
    mean_shift = getdfcol(df,3)
    worst_shift = getdfcol(df,7)
    meta_fit_resfile = "slope_routability_fit_results.csv"
    meta_fit_result_lines = ["function:" + meta_param_func.name,"what is fitted, arg1, arg2, arg3, MSE"]
    if _arb:
        errorfunc = lambda vars,x,data : logfunc(x, vars[0], vars[1], vars[2]) - data
        RoutabilitypointArbitrary = least_squares(errorfunc, x0=meta_fit_vars['start'], args=(mesh_area_values, initial_shift), method="lm")
        ax.scatter(mesh_area_values, initial_shift, c=initial_col, label='initial pathlist')
        ax.plot(INTERP_MESH, logfunc(INTERP_MESH, *list(RoutabilitypointArbitrary.x)), c=black, linestyle="--")


    if _best:
        RoutabilitypointBest = least_squares(errorfunc, x0=meta_fit_vars['start'], args=(mesh_area_values, best_shift), method="lm")
        ax.scatter(mesh_area_values, best_shift, c=best_col, label='after permutation')
        ax.plot(INTERP_MESH, logfunc(INTERP_MESH, *list(RoutabilitypointBest.x)), c=black, linestyle="--")

    if _mean:
        ax.scatter(mesh_area_values, mean_shift, c=mean_col, label='discovered shift mean')
        RoutabilitypointMean = least_squares(errorfunc, x0=meta_fit_vars['start'], args=(mesh_area_values, mean_shift), method="lm")
        ax.plot(INTERP_MESH, logfunc(INTERP_MESH, *list(RoutabilitypointMean.x)), c=black, linestyle="--")

    if _worst:
        ax.scatter(mesh_area_values, worst_shift, c=worst_col, label='discovered shift worst')
        RoutabilitypointWorst = least_squares(errorfunc, x0=meta_fit_vars['start'], args=(mesh_area_values, worst_shift), method="lm")
        ax.plot(INTERP_MESH, logfunc(INTERP_MESH, *list(RoutabilitypointWorst.x)), c=black, linestyle="--")

    plt.xlabel("mesh area")
    plt.ylabel("routability point")
    meta_fit_result_lines.append(",".join(["initial routabilitypoint", str(RoutabilitypointArbitrary.x[0]), str(RoutabilitypointArbitrary.x[1]), str(RoutabilitypointArbitrary.x[2]), str(np.mean(np.power(RoutabilitypointArbitrary.fun, 2)))]))
    meta_fit_result_lines.append(",".join(["routabilitypoint after permutation", str(RoutabilitypointBest.x[0]), str(RoutabilitypointBest.x[1]), str(RoutabilitypointBest.x[2]), str(np.mean(np.power(RoutabilitypointBest.fun, 2)))]))
    ax.text(4900, 30, eval(format_found_params_inside_parenthesis(*list(RoutabilitypointArbitrary.x))), fontsize=12)
    ax.text(4900, 57.3, eval(format_found_params_inside_parenthesis(*list(RoutabilitypointBest.x))), fontsize=12)

    print(*RoutabilitypointArbitrary.x)
    print(*RoutabilitypointBest.x)


    plt.legend(loc=2)
    plt.show()

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    initial_slope = getdfcol(df,2)
    best_slope = getdfcol(df,6)
    mean_slope = getdfcol(df,4)
    worst_slope = getdfcol(df,8)
    if _arb:
        errorfunc = lambda vars,x,data : slope_adapted_logfunc(x, vars[0], vars[1], vars[2]) -data
        OptimizeResultSlopeArbitrary = least_squares(errorfunc, x0=meta_fit_vars['start_s'], args=(mesh_area_values, initial_slope), method="lm")
        slope_arbitrary_corrected = correct_slopefunc(mesh_area_values, *list(OptimizeResultSlopeArbitrary.x))
        ax.scatter(mesh_area_values, initial_slope, c=initial_col, label='initial pathlist')
        ax.plot(INTERP_MESH, slope_adapted_logfunc(INTERP_MESH, *slope_arbitrary_corrected), c=black, linestyle="--")

    if _best:
        OptimizeResultSlopeBest = least_squares(errorfunc, x0=meta_fit_vars['start_s'], args=(mesh_area_values, best_slope), method="lm")
        slope_best_corrected = correct_slopefunc(mesh_area_values, *list(OptimizeResultSlopeBest.x))
        ax.scatter(mesh_area_values, best_slope, c=best_col, label='after permutation')
        ax.plot(INTERP_MESH, slope_adapted_logfunc(INTERP_MESH, *slope_best_corrected), c=black, linestyle="--")

    if _mean:
        OptimizeResultSlopeMean = least_squares(errorfunc, x0=meta_fit_vars['start_s'], args=(mesh_area_values, mean_slope), method="lm")
        slope_mean_corrected = correct_slopefunc(mesh_area_values, *list(OptimizeResultSlopeMean.x))
        ax.scatter(mesh_area_values, mean_slope, c=mean_col, label='discovered mean sequence')
        ax.plot(INTERP_MESH, slope_adapted_logfunc(INTERP_MESH, *slope_mean_corrected), c=black, linestyle="--")

    if _worst:
        OptimizeResultSlopeWorst = least_squares(errorfunc, x0=meta_fit_vars['start_s'], args=(mesh_area_values, worst_slope), method="lm")
        slope_worst_corrected = correct_slopefunc(mesh_area_values, *list(OptimizeResultSlopeWorst.x))
        ax.scatter(mesh_area_values, worst_slope, c=worst_col, label='discovered worst sequence')
        ax.plot(INTERP_MESH, slope_adapted_logfunc(INTERP_MESH, *slope_worst_corrected), c=black, linestyle="--")

    plt.xlabel("mesh area")
    plt.ylabel("slope")

    print(*slope_arbitrary_corrected)
    print(*slope_best_corrected)

    ax.text(3400, .18, eval(format_found_params_inside_parenthesis(*slope_arbitrary_corrected)), fontsize=12)
    ax.text(3400, .22, eval(format_found_params_inside_parenthesis(*slope_best_corrected)), fontsize=12)
    meta_fit_result_lines.append(",".join(["initial slope", str(slope_arbitrary_corrected[0]), str(slope_arbitrary_corrected[1]), str(slope_arbitrary_corrected[2]), str(np.mean(np.power(OptimizeResultSlopeArbitrary.fun, 2)))]))
    meta_fit_result_lines.append(",".join(["slope after permutation", str(slope_best_corrected[0]), str(slope_best_corrected[1]), str(slope_best_corrected[2]), str(np.mean(np.power(OptimizeResultSlopeBest.fun, 2)))]))
    with open(meta_fit_resfile, "w+") as f:
        f.write("\n".join(meta_fit_result_lines))
    plt.legend(loc=1)
    plt.show()



if __name__ == '__main__':
    config_dict = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--routings_per_pathlist", help="the number of different orders each pathlist should be routed, the paper uses 200", type=int, default=config_dict['routings_per_pathlist'])
    args = parser.parse_args()
    N = args.routings_per_pathlist

    meta_function = "logfunc"
    meta_fit("area", "inverted_logistic", meta_function, _arb=True, _mean=False, _best=True, _worst=False)
