
This repository contains the experimental setup of the paper:

Reitze Jansen, Yannick Vinkesteijn, and Daan van den Berg. 2020. On the solvability of routing multiple point-to-point paths in manhattan meshes. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion (GECCO '20). Association for Computing Machinery, New York, NY, USA, 1685–1689. DOI:https://doi.org/10.1145/3377929.3398098

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



# referencing this work:

if you use this repository please cite this work:

@inproceedings{10.1145/3377929.3398098,
author = {Jansen, Reitze and Vinkesteijn, Yannick and van den Berg, Daan},
title = {On the Solvability of Routing Multiple Point-to-Point Paths in Manhattan Meshes},
year = {2020},
isbn = {9781450371278},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3377929.3398098},
doi = {10.1145/3377929.3398098},
abstract = {The solvability of multiple path routing problems in 3D Manhattan meshes is greatly influenced by the order in which the paths are processed. Unsolvable instances can readily be made solvable by simply changing the sequence of paths to be routed, and the inverse also holds. Our results on square meshes with 100 randomly placed terminals show that the routability of an instance can be accurately guessed a priori from its characteristics only. Furthermore, the attainable routability from random sequence change can also be accurately guessed, and a tight scaling relation suggests these results hold for a broad range of instance sizes. For our particulars, the number of routable paths can increase as much as 73%, just by changing the order of processing.},
booktitle = {Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
pages = {1685–1689},
numpages = {5},
keywords = {phase transition, 3D routing, solvability, instance hardness, a* routing, scaling law},
location = {Canc\'{u}n, Mexico},
series = {GECCO '20}
}
