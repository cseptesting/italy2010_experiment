
conda create -n float_italy -y
conda activate float_italy
conda install -c conda-forge pycsep -y
git clone https://github.com/cseptesting/floatcsep.git  --branch=main --depth=1
pip install -e floatcsep/.
git clone https://github.com/SCECcode/pycsep.git --branch=master --depth=1
pip install -e pycsep/.
pip install -r extra_reqs.txt