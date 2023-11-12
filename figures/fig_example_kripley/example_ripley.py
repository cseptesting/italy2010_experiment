import matplotlib.pyplot as plt
import numpy as np
import os
import numpy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import floatcsep.utils
import csep
import floatcsep
from floatcsep import experiment
import seaborn as sns
import cartopy
import rioxarray as rxr
sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
plt.rcParams.update({
    'xtick.bottom': True,
    'ytick.left': True,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9})


def read_hdf5(fname):

    with h5py.File(fname, 'r') as db:
        parsed_results = dict.fromkeys(db.keys())
        for key, val in db.items():
            if key != 'name':
                parsed_results[key] = val[:]
            else:
                parsed_results[key] = val

    return parsed_results


def plot_forecasts_sims():

    cfg = '../../src/total/config.yml'
    exp = experiment.Experiment.from_yml(cfg)
    exp.set_tasks()
    exp.stage_models()

    # CATALOG
    prop_forecast_a = {'grid_labels': True,
                       'borders': False,
                       'region_border': False,
                       'coastline': False,
                       'feature_lw': 0.5,
                       'cmap': 'magma',
                       'alpha_exp': 0.3,
                       'clim': [-6, -1],
                       'basemap': None,
                       'projection': cartopy.crs.Mercator()}

    prop_forecast_b = {'grid_labels': True,
                       'borders': False,
                       'region_border': False,
                       'coastline': False,
                       'feature_lw': 0.5,
                       'cmap': 'magma',
                       'alpha_exp': 0.3,
                       'clim': [-6, -1],
                       'basemap': None,
                       'projection': cartopy.crs.Mercator()}

    cat_props = {'markersize': 12,
                 'markercolor': 'red',
                 'mag_scale': 1,
                 'legend': False,
                 'coastline': False,
                 'basemap': None,
                 'alpha': 0.5}

    sim_props = {'markersize': 8,
                 'markercolor': 'lightgrey',
                 'mag_scale': 1,
                 'coastline': False,
                 'legend': False,
                 'basemap': None,
                 'alpha': 0.8}
    #
    cat = csep.load_catalog('../../catalogs/catalog_ml.json')
    region = csep.regions.italy_csep_region()
    cat.filter_spatial(region, in_place=True)
    projection = cartopy.crs.Mercator()

    counts = cat.spatial_counts()
    A = np.zeros(region.idx_map.shape)
    A[ A == 0] = np.nan
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isnan(region.idx_map[i,j]):
                A[i, j] = counts[int(region.idx_map[i,j])]

    cat_ind = cat.get_spatial_idx()
    Events = np.array([[i, j] for i, j in zip(cat.get_longitudes(), cat.get_latitudes())])

    model_a = exp.get_model('HAZFX_BPT')
    model_b = exp.get_model('TripleS-CSI')

    tstring = floatcsep.utils.timewindow2str(exp.timewindows[0])
    model_a.create_forecast(tstring)
    model_b.create_forecast(tstring)

    forecast_a = model_a.get_forecast(tstring)
    forecast_b = model_b.get_forecast(tstring)

    n_sim = 1
    numpy.random.seed(25)   # 4, 25

    #################
    # Forecast A
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

    raster_fn = 'basemap_3857.tif'
    rds = rxr.open_rasterio(raster_fn)
    os.makedirs('forecasts', exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=projection)
    rds.plot.imshow(ax=ax, transform=projection, add_labels=False)
    Ax = forecast_a.plot(ax=ax, plot_args=prop_forecast_a)
    Ax = cat.plot(ax=Ax, plot_args=cat_props)
    sim_props['title'] = forecast_a.name
    sim_cat.plot(ax=Ax, plot_args=sim_props, show=True, extent=region.get_bbox())
    pts = region.tight_bbox()
    ax.plot(pts[:, 0], pts[:, 1], lw=1, color='black',
            transform=cartopy.crs.PlateCarree())
    plt.savefig('forecast_a.png', dpi=300)
    plt.show()

    #
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
    tuple = [(1, 1, i[1], i[0], 5,10) for i in events_sim]
    sim_cat = csep.catalogs.CSEPCatalog(data=tuple)
    sim_cat.region = region

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=projection)
    rds.plot.imshow(ax=ax, transform=projection, add_labels=False)
    Ax = forecast_b.plot(ax=ax, plot_args=prop_forecast_b)
    Ax = cat.plot(ax=Ax, plot_args=cat_props)
    sim_props['title'] = forecast_b.name
    sim_cat.plot(ax=Ax, plot_args=sim_props, show=True, extent=region.get_bbox())
    pts = region.tight_bbox()
    ax.plot(pts[:, 0], pts[:, 1], lw=1, color='black',
            transform=cartopy.crs.PlateCarree())
    plt.savefig('forecast_b.png', dpi=300)
    plt.show()


def plot_l_funcs(alpha=0.05, figsize=(6, 4)):

    ripley_path = os.path.join('../../src', 'ripley')

    results_hazfx = read_hdf5(os.path.join(ripley_path,
                                           'results',
                                           f'K_HAZFX_BPT.hdf5'))
    results_hazfx['name'] = 'HAZFX-BPT'
    results_triples = read_hdf5(os.path.join(ripley_path,
                                           'results',
                                           f'K_TripleS-CSI.hdf5'))
    results_triples['name'] = 'TripleS-CSI'
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True,
                            sharey=True)
    for ax, result in zip(axs, [results_hazfx, results_triples]):
        name = result.get('name', 'None')
        r = result['r']
        l_sims = result['l_sims'][::7]
        l_cat = result['l_cat']
        L_avg = result['l_mean']
        L_up = result[f'l_{1-alpha/2:.2f}']
        L_down = result[f'l_{alpha/2:.2f}']
        L_over = result[f'l_over_{alpha:.2f}']

        for i in l_sims:
            sns.lineplot(ax=ax, x=r, y=i, lw=0.05, color='black')
        sns.lineplot(ax=ax, x=r, y=l_cat, color='r', label='Observed catalog',
                     legend=False)
        sns.lineplot(ax=ax, x=r, y=L_avg, label='Sim. average',
                     legend=False)
        ax.fill_between(r, L_down, L_up, color='gray', alpha=0.4,
                        label=r'Sim. envelope ($\gamma=%.2f$)' % (1 - alpha))
        ax.fill_between(r, L_up, l_cat, where=l_cat >= L_up,
                        color='red', alpha=0.3, label=r'Envelope distance')

        ax.set_xlim([None, np.max(r)])
        ax.set_xlabel(r"Bandwidth $r~~\mathrm{[km]}$", fontsize=15)
        ax.set_ylim([0, 950])
        ax.set_title(name, fontsize=18, loc='left')
    axs[0].set_ylabel(r"$L(r) = \sqrt{\frac{K(r)}{\pi}}$", fontsize=18)
    axs[0].legend(loc=2, fontsize=12)

    plt.savefig(f'L_models.png', dpi=300, facecolor=(0, 0, 0, 0))
    plt.show()


if __name__ == '__main__':
    plot_forecasts_sims()
    plot_l_funcs()


