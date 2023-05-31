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