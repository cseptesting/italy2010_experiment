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


cfg = '../config_test.yml'
exp = experiment.Experiment.from_yml(cfg)
exp.set_tests()
exp.set_models()



#################### CATALOG
prop_forecast_a = {'grid_labels': True,
                   'borders': True,
                   'feature_lw': 0.5,
                   'cmap': 'magma',
                   'alpha_exp': 0.3,
                   'clim': [-7, 0],
                   'basemap': 'ESRI_imagery',
                   'projection': cartopy.crs.Mercator()}

prop_forecast_b = {'grid_labels': True,
                   'borders': True,
                   'feature_lw': 0.5,
                   'cmap': 'magma',
                   'alpha_exp': 0.3,
                   'clim': [-7, 0],
                   'basemap': 'ESRI_imagery',
                   'projection': cartopy.crs.Mercator()}



cat_props = {'markersize': 12,
             'markercolor': 'red',
             'mag_scale': 1,
             'legend': False,
             'basemap': None,
             'alpha': 0.5}

sim_props = {'markersize': 8,
             'markercolor': 'lightgrey',
             'mag_scale': 1,
             'legend': False,
             'basemap': None,
             'alpha': 0.8}
#
cat = csep.load_catalog('../catalog_ml.json')
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



model_a = exp.get_model('HAZFX_BPT')
model_b = exp.get_model('TripleS-CSI')

tstring = fecsep.utils.timewindow2str(exp.timewindows[0])
model_a.create_forecast(tstring)
model_b.create_forecast(tstring)

forecast_a = model_a.get_forecast(tstring)
forecast_b = model_b.get_forecast(tstring)


# forecast_a.plot(plot_args=prop_forecast_a)
# plt.show()
# forecast_b.plot(plot_args=prop_forecast_b)
# plt.show()



n_sim = 1



numpy.random.seed(25)   # 4, 25

#################
## Forecast A
#################

rates_eqk = forecast_a.spatial_counts()[cat_ind]
observed_data = cat.spatial_counts()
n_obs = numpy.sum(observed_data)
n_fore = numpy.sum(forecast_a.spatial_counts())
scale = n_obs / n_fore
expected_forecast_count = int(n_obs)
num_events_to_simulate = expected_forecast_count
sampling_weights = numpy.cumsum(forecast_a.spatial_counts().ravel()) / \
                   numpy.sum(forecast_a.spatial_counts())
sim_fore = numpy.zeros(sampling_weights.shape)
sim = csep.core.poisson_evaluations._simulate_catalog(num_events_to_simulate, sampling_weights, sim_fore)
ind_sim = np.where(sim != 0)[0]  #todo fix 2 events per cell
events_sim = np.array([forecast_a.get_longitudes()[ind_sim], forecast_a.get_latitudes()[ind_sim]]).T
rates_sim = forecast_a.data[ind_sim]
tuple = [ (1, 1, i[1], i[0], 5,10) for i in events_sim]
sim_cat = csep.catalogs.CSEPCatalog(data=tuple)
sim_cat.region=region

Ax = forecast_a.plot(plot_args=prop_forecast_a)
Ax = cat.plot(ax=Ax, plot_args=cat_props)
sim_props['title'] = forecast_a.name
sim_props['filename'] = 'forecast_a'
sim_cat.plot(ax=Ax, plot_args=sim_props, show=True, extent=region.get_bbox())
plt.show()


numpy.random.seed(43)   # 27, 43
##################
### Forecast B
##################
rates_eqk = forecast_b.spatial_counts()[cat_ind]
observed_data = cat.spatial_counts()
n_obs = numpy.sum(observed_data)
n_fore = numpy.sum(forecast_b.spatial_counts())
scale = n_obs / n_fore
expected_forecast_count = int(n_obs)
num_events_to_simulate = expected_forecast_count
sampling_weights = numpy.cumsum(forecast_b.spatial_counts().ravel()) / \
                   numpy.sum(forecast_b.spatial_counts())
sim_fore = numpy.zeros(sampling_weights.shape)
sim = csep.core.poisson_evaluations._simulate_catalog(num_events_to_simulate, sampling_weights, sim_fore)
ind_sim = np.where(sim != 0)[0]  #todo fix 2 events per cell
events_sim = np.array([forecast_b.get_longitudes()[ind_sim], forecast_b.get_latitudes()[ind_sim]]).T
rates_sim = forecast_b.data[ind_sim]
tuple = [ (1, 1, i[1], i[0], 5,10) for i in events_sim]
sim_cat = csep.catalogs.CSEPCatalog(data=tuple)
sim_cat.region=region

Ax = forecast_b.plot(plot_args=prop_forecast_b)
Ax = cat.plot(ax=Ax, plot_args=cat_props)
sim_props['title'] = forecast_b.name
sim_props['filename'] = 'forecast_b'
sim_cat.plot(ax=Ax, plot_args=sim_props, show=True, extent=region.get_bbox())
plt.show()

#



#
# Ax = models[model_i].plot(plot_args=plot_properties)
# sim_props['title'] = models[model_i].name
# sim_props['filename'] = paths.figs + '/sims/' + models[model_i].name+'_%i.png'%i
# Ax1 = cat.plot(ax=Ax, plot_args=cat_props)
# sim_cat.region=region
# sim_cat.plot(ax=Ax,plot_args=sim_props, show=True)
#
#
