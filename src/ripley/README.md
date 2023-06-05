# Floating Experiment - Italy time independent

## K-ripley analysis

The computational environment can be reproduced using conda, R and binary libraries or simply using a Dockerfile (recommended)


### Running without Docker

Python should be installed with the following packages:
* `pycsep`==0.6.1
* `floatcsep`==0.1.2
* `rpy2`==3.5.12

R should be installed with the following versions

* `R`>=4.0.4
* `spatstat.geom`==2.2-0
* `spatstat.core`==2.2-0

To install R and its dependencies
```shell
sudo apt-get update
sudo apt-get install r-base
Rscript -e "nproc <- as.integer(commandArgs(trailingOnly = TRUE)[1]); \
            install.packages('remotes', Ncpus=nproc) ; \
            require(remotes); \
            install_version('spatstat.core', version = '2.2-0', Ncpus=nproc); \
            install_version('spatstat.geom', version = '2.2-0', Ncpus=nproc)" $(nproc)
```

To set up the conda computational environment, simply
```
conda env create -f environment.yaml
conda activate kripley
```

To run the script and generate the results type:
```shell
python main.py
```


### Docker-Build (recommended)

Having Docker installed (see https://docs.docker.com/engine/install/ubuntu/  and  https://docs.docker.com/engine/install/linux-postinstall/), run the following command to rebuild the full docker image.

```shell
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t kripley .
```

### Docker run

Once the image has been built, run the script as

```shell
 docker run --rm --volume $PWD:/usr/src/kripley:rw kripley 
```

### Docker interactive run

If extra granularity is needed, the interactive steps are:

```shell
 docker run -it --rm --volume $PWD:/usr/src/kripley:rw kripley /bin/bash
conda activate kripley
python k_function.py
```