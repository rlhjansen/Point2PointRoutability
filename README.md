
# environment setup:

to install dependencies we recommend installing via anaconda:

    conda env create -f routability_environment.yml

    conda activate routability


# experiment setup
setup the experiment (meshes, pathlists):

    python setup_experiment.py


run the experiment (this takes 2 days and 14 hours on our setup, with multiprocessing on 10 cpus):

    python run_experiment.py

this can be ran with an optional argument, which determines the number of routing permutations per pathlists; default (in the paper is 200). example usage which would reproduce the experiment setup explicitly:

    python run_experiment.py --routings_per_pathlist 200


# recreating plots:

After running the experiment, plots can be obtained by calling:

    python make_plots.py

this wil output files with the ratios of routability for each net, both initially, as well as after permutation, as well as some other stats.
* "compare_routability_best_of_N.csv" gives an overview of routability ratios per netlistlength, per meshsize. this file contains the scattered values in the 3x3 window in the paper.
* "routability_fits_per_meshsize.csv" gives an overview of parameters of the inverted logistic function we fit per meshsize.
* "slope_routability_fit_results.csv" gives an overview of parameters of the logarithmic function we fit over routability-points & slopes per meshsize.

furthermore, it will show plots displaying results like in the paper.

just like running the experiment, this can be called with the optional argument:

    python make_plots.py --routings_per_pathlist *N*

Here N is an integer that determines the number of permutations.
this can be done to get an insight in how the improvement gained bij routing more times increases as the number of permutations increases; the default setting is the number permutations of the experiments ran, but can be any number *between 1 and that number* (which is saved in config.txt in case it is forgotten).
