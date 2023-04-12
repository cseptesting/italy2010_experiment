# Floating Experiment - Italy time independent


## Install
create conda venv fe_italy_ti
```
conda create -n fe_italy 
```

```
git clone https://github.com/cseptesting/fecsep.git  --branch=main --depth=1
cd fecsep
conda env update --file environment.yml --prune
pip install -e .
```


If the latest `pycsep` version is needed
```
git clone https://github.com/SCECcode/pycsep.git --branch=master --depth=1
cd pycsep
pip install -e .
```