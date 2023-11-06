# Floating Experiment - Italy 2010 Forecasting Experiment


## Overview


This repository contains the complete Italy 2010 Experiment  definition (Schorlemmer et al., 2010), input data, methods, artifacts and results (Iturrieta et al., in-prep).

The experiment was conducted using the [floatCSEP](https://github.com/cseptesting/floatcsep) and [pyCSEP](https://github.com/SCECcode/pycsep) for most of the testing routines, along with `R` for the spatial statistics analyses.


## Install

Create working environment
```
conda create -n float_italy 
conda activate float_italy
```

Install floatCSEP
```
conda install -c conda-forge floatcsep=0.1.4
```


## Run

The experiment source code is divided into three:

* The experiment of a single, ten-year time window. Here are found all the consistency tests, rankings and comparison tests. The source code is located in the `src/total` folder. Here, the code can be run as:

    ```
    cd src/total
    floatcsep run config.yml
    ```
    or the results reproduced with:
    ```
    cd src/total/results
    floatcsep reproduce repr_config.yml
    ```

* The experiment of multiple, one-year cumulative time windows, which contain the sequential rankings. The source code is located in the `src/sequential` folder. The code can be run as:

    ```
    cd src/sequential
    floatcsep run config.yml
    ```
  or the results reproduced with:
    ```
    cd src/sequential/results
    floatcsep reproduce config.yml
    ```


* The analysis of spatial performance using second-order statistics, i.e. Ripley-type metrics. It can be run as

    ```
    cd src/ripley
    python main.py
    ```

    Moreover, it can be reproduced using a Docker container. Docker can be installed following the instructions from [https://docs.docker.com/engine/install/]. Please check the instructions in `src/ripley/README.md` for more details.

## Figure generation

The figures of the `Iturrieta et al., in-prep` manuscript can be reproducing using the source code found in the folder `figures/`. These codes should be run once the corresponding results have been calculated.    
