import matplotlib.pyplot as plt
import numpy as np
import os
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from csep.utils.comcat import search
import fecsep.utils
import csep
import fecsep
from fecsep import experiment
import itertools
from datetime import datetime
import cartopy


cfg = '../../runs/total/config.yml'
exp = experiment.Experiment.from_yml(cfg)
exp.set_tests()
exp.set_models()

#################### CATALOG
fc_prop = {'grid_labels': True,
                   'feature_lw': 0.5,
                   'cmap': 'magma',
                   'alpha_exp': 0.3,
                   'clim': [-7, 0],
           'clabel': r'$\log_{10}\mu\,(M>5)$',
           'clabel_fontsize': 14,
                   'basemap': 'ESRI_imagery',
                   'projection': cartopy.crs.Mercator()}

cat_prop = {'markersize': 1,
             'markercolor': 'black',
             'mag_scale': 1,
             'legend': False,
             'basemap': None,
              'linecolor': 'white',
             'alpha': 0.5}

cat = csep.load_catalog('catalog.json')
region = csep.regions.italy_csep_region()
cat.filter_spatial(region, in_place=True)


counts = cat.spatial_counts()
A = np.zeros(region.idx_map.shape)
A[ A == 0] = np.nan
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if not np.isnan(region.idx_map[i,j]):
            A[i, j] = counts[int(region.idx_map[i,j])]

cat_ind = cat.get_spatial_idx()
Events = np.array([[i, j] for i,j in zip(cat.get_longitudes(), cat.get_latitudes())])
tstring = fecsep.utils.timewindow2str(exp.timewindows[0])

for model in exp.models:
    model_name = model.name
    model.create_forecast(tstring)
    forecast = model.get_forecast(tstring)
    Ax = forecast.plot(plot_args=fc_prop)
    cat_prop['title'] = model_name
    cat_prop['title_size'] = 15
    Ax = cat.plot(ax=Ax, plot_args=cat_prop)
    plt.savefig(f'{model_name}.png', dpi=200)
    plt.show()
