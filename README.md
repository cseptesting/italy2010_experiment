# Floating Experiment - Italy time independent


## Install
create conda venv fe_italy_ti
```
conda create -n fe_italy 
```

```
git clone git@git.gfz-potsdam.de:csep-group/fecsep.git --branch=dev --depth=1
cd fecsep
conda env update --file environment.yml --prune
pip install -e .
git clone https://github.com/SCECcode/pycsep.git --branch=master --depth=1
cd fecsep
```