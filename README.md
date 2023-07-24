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
conda install -c conda-forge pycsep==0.6.2
```

Install floatCSEP
```
pip install floatcsep==0.1.3
```


If the latest `pycsep` and `floatcsep` versions are needed
```
git clone https://github.com/SCECcode/pycsep.git --depth=1
git clone https://github.com/cseptesting/floatcsep.git --depth=1
pip install -e ./pycsep/
pip install -e ./floatcsep/
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