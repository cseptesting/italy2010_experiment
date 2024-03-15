# Floating Experiment - Italy 2010 Forecasting Experiment

<img src="https://i.postimg.cc/Bb60rVQP/CSEP2-Logo-CMYK.png" width="320">

## Overview


This repository contains the complete Italy 2010 Experiment  definition (Schorlemmer et al., 2010), input data (catalog and forecasts), methods, artifacts and results (Iturrieta et al., 2024) to be reproduced. The experiment was conducted using the [floatCSEP](https://github.com/cseptesting/floatcsep) and [pyCSEP](https://github.com/SCECcode/pycsep) for most of the testing routines, along with `R` to perform the spatial statistics analyses.


## Install

* Set up ``Python`` and `Conda` using the [Miniforge](https://github.com/conda-forge/miniforge) or [MiniConda](https://docs.conda.io/en/latest/miniconda.html) distributions, following the instructions in the links.
* Create the working environment
  ```bash
  mamba env create -f environment.yml -y
  ```
  replace `mamba` with `conda` if `MiniConda` rather than `Miniforge` was installed. 
  
* Activate the environment
  ```bash
  conda activate float_italy
  ```


## Run

The numerical results should be generated first, and after the Iturrieta et al., (2024) figures can be reproduced. The experiment source code is divided into three:

* The experiment of a single, ten-year time window. Here are found all the consistency tests, rankings and comparison tests. The source code is located in the `src/total` folder. Here, the results can be obtained from a `bash` console as:

    ```bash
    cd src/total
    floatcsep run config.yml
    ```
    or the exact results reproduced with:
    ```bash
    cd src/total/results
    floatcsep reproduce repr_config.yml
    ```

* The experiment of multiple, one-year cumulative time windows, which contain the sequential rankings. The source code is located in the `src/sequential` folder. The code can be run as:

    ```bash
    cd src/sequential
    floatcsep run config.yml
    ```
  or the results reproduced with:
    ```bash
    cd src/sequential/results
    floatcsep reproduce config.yml
    ```


* The analysis of spatial performance using second-order statistics, i.e. Ripley-type metricsm, can be reproduced using a Docker container. Docker can be installed following the instructions from [https://docs.docker.com/engine/install/]. Please check the instructions in `src/ripley/README.md` for more details.

## Figure generation

The figures of the **Iturrieta et al., 2024** manuscript can be reproducing using the source code found in the folder `figures/`. These codes should be run once the corresponding results have been calculated.   

## References

* Schorlemmer, Danijel, et al. "Setting up an earthquake forecast experiment in Italy." Annals of Geophysics (2010).
* Taroni, Matteo, et al. "Prospective CSEP evaluation of 1‐day, 3‐month, and 5‐yr earthquake forecasts for Italy." Seismological Research Letters 89.4 (2018): 1251-1261.
* Iturrieta, Pablo, et al. "Evaluating a decade-long forecasting experiment in Italy." Seismological Research Letters (2024).