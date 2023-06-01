`# Floating Experiment - Italy time independent
`

## Install

Create working environment
```
conda create -n float_italy 
conda activate float_italy
```

Install pyCSEP
```
conda install -c conda-forge pycsep 
```

Install floatCSEP
```
git clone https://github.com/cseptesting/floatcsep.git  --branch=main --depth=1
cd fecsep
pip install -e .
```


If the latest `pycsep` commit is needed
```
git clone https://github.com/SCECcode/pycsep.git --branch=master --depth=1
cd pycsep
pip install -e .
```


## Docker

### Build

```
docker build \
--build-arg USERNAME=$USER \
--build-arg USER_UID=$(id -u) \
--build-arg USER_GID=$(id -g) \
-t floatitaly .
```
### Run

```
docker run -it --rm --volume $PWD:/usr/src/float_italy:rw floatitaly /bin/bash

```