# Transitions in optimal adaptive strategies for populations in fluctuating environments

This repository contains the source code associated with the manuscript

Mayer, Mora, Rivoire, Walczak : [Transitions in optimal adaptive strategies for populations in fluctuating environments](), Arxiv 2017

It allows reproduction of all numerical results reported in the manuscript.

## Installation requirements

The code uses Python 2.7+.

A number of standard scientific python packages are needed for the numerical simulations and visualizations. An easy way to install all of these is to install a Python distribution such as [Anaconda](https://www.continuum.io/downloads). The file `environment.yml` contains a list of the relevant packages in a format understood by Anaconda.

- [numpy](http://github.com/numpy/numpy/)
- [scipy](https://github.com/scipy/scipy)
- [pandas](http://github.com/pydata/pandas)
- [matplotlib](http://github.com/matplotlib/matplotlib)

Additionally the code also relies on these packages:

- [scipydirect](http://github.com/andim/scipydirect/)
- [noisyopt](http://github.com/andim/noisyopt)
- [palettable](http://github.com/jiffyclub/palettable)

And optionally for nicer progress output install:

- [pyprind](http://github.com/rasbt/pyprind)

## Running the code

The time stepping of the population dynamics is accelerated by a Cython module, which needs to be compiled first. To compile it run `make cython` in the `lib` directory. In the directories for the figures about the results in correlated environments launch `make run` followed by `make agg` to produce the underlying data. We provide both Jupyter notebooks with additional explanatory comments and plain python files for generating the figures.

Note: As the simulations are stochastic you will not get precisely equivalent plots.

## Contact

If you run into any difficulties running the code, feel free to contact us at `andisspam@gmail.com`.

## License

The source code is freely available under an MIT license.
