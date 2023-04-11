import matplotlib.pyplot as plt
import numpy as np
import os
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

import fecsep.utils
from pycsep import csep
import fecsep
from fecsep import experiment
import itertools

import cartopy

#################### CATALOG
plot_properties = {'grid_labels': True,
                   'borders': True,
                   'feature_lw': 0.5,
                   'basemap': 'ESRI_terrain',
                   'cmap': 'rainbow',
                   'alpha_exp': 0.8,
                   'clim' : [-8, 0],
                   'projection': cartopy.crs.Mercator()}
cat_props =  {'markersize': 3,
              'markercolor': 'red',
              'alpha': 0.6}
sim_props =  {'markersize': 3,
              'markercolor': 'blue',
              'alpha': 0.6}

cat = csep.load_catalog('../catalog.json')
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

cfg = '../config.yml'
exp = experiment.Experiment.from_yml(cfg)
exp.set_tests()
exp.set_models()

model_a = exp.get_model('HAZFX_BPT')
model_b = exp.get_model('HRSS-m1')

exp.time_config
tstring = fecsep.utils.timewindow2str(exp.timewindows[0])
model_a.create_forecast(tstring)
model_b.create_forecast(tstring)

forecast_a = model_a.get_forecast(tstring)
forecast_b = model_b.get_forecast(tstring)
#

forecast_b.plot()

#
# range_models = range(1,2)
# savefig =  True
# n_sim = 100
#
# T_x = []
# T = []
# for model_i in range_models:
#     forecast_data = models[model_i].spatial_counts()
#     rates_eqk = forecast_data[cat_ind]
#     plot_properties['title'] = models[model_i].name
#
#     forecast_data = models[model_i].spatial_counts()
#     observed_data = cat.spatial_counts()
#     n_obs = numpy.sum(observed_data)
#     n_fore = numpy.sum(forecast_data)
#     scale = n_obs / n_fore
#     expected_forecast_count = int(n_obs)
#     num_events_to_simulate = expected_forecast_count
#     sampling_weights = numpy.cumsum(forecast_data.ravel()) / numpy.sum(forecast_data)
#     #
#     Ax = models[model_i].plot(plot_args=plot_properties)
#     cat_props['title'] = models[model_i].name
#     cat_props['filename'] = paths.figs + '/sims/' + models[model_i].name+'_CAT.png'
#     Ax1 = cat.plot(ax=Ax, plot_args=cat_props)
#
#     for i in range(n_sim):
#
#         sim_fore = numpy.zeros(sampling_weights.shape)
#         sim = csep.core.poisson_evaluations._simulate_catalog(num_events_to_simulate, sampling_weights, sim_fore)
#         ind_sim = np.where(sim != 0)[0]  #todo fix 2 events per cell
#         events_sim = np.array([models[model_i].get_longitudes()[ind_sim], models[model_i].get_latitudes()[ind_sim]]).T
#         rates_sim = forecast_data[ind_sim]
#         tuple = [ (1, 1, i[1], i[0], 5,10) for i in events_sim]
#         sim_cat = csep.catalogs.CSEPCatalog(data=tuple)
#         Ax = models[model_i].plot(plot_args=plot_properties)
#         sim_props['title'] = models[model_i].name
#         sim_props['filename'] = paths.figs + '/sims/' + models[model_i].name+'_%i.png'%i
#         Ax1 = cat.plot(ax=Ax, plot_args=cat_props)
#         sim_cat.region=region
#         sim_cat.plot(ax=Ax,plot_args=sim_props, show=True)
#
#
