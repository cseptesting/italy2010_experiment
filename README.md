`# Floating Experiment - Italy time independent
`

## Install

Create working environment
```
conda create -n fe_italy 
conda activate -n fe_italy
```

Install pyCSEP
```
conda install -c conda-forge pycsep 
```

Install fxCSEP
```
git clone https://github.com/cseptesting/fecsep.git  --branch=main --depth=1
cd fecsep
pip install -e .
```


If the latest `pycsep` commit is needed
```
git clone https://github.com/SCECcode/pycsep.git --branch=master --depth=1
cd pycsep
pip install -e .
```