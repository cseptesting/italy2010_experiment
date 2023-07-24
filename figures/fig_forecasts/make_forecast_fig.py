import matplotlib.pyplot as plt
import rioxarray as rxr
import numpy as np
import csep
import os
import floatcsep
from floatcsep import experiment
import cartopy
import seaborn
seaborn.set_style("white", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
plt.rcParams.update({'xtick.bottom': True, 'axes.labelweight': 'bold',
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 9})


cfg = '../../src/total/config.yml'


exp = experiment.Experiment.from_yml(cfg)
projection = cartopy.crs.GOOGLE_MERCATOR

fc_prop = {'grid_labels': True,
           'feature_lw': 0.2,
           'linecolor': 'black',
           'basemap': None,
           'title': None,
           'include_cbar': False,
           'cmap': 'magma',
           'alpha_exp': 0.3,
           'clim': [-7, 0],
           'clabel': r'$\log_{10}\mu\,(M>5)$',
           'clabel_fontsize': 14,
           'projection': projection}

cat_prop = {'markersize': 6,
            'markeredgecolor': 'black',
            'markercolor': 'white',
            'mag_scale': 7.5,
            'title': None,
            'legend': False,
            'basemap': None,
            'linewidth': 0.1,
            'alpha': 0.3}

cat = csep.load_catalog('catalog.json')
region = csep.regions.italy_csep_region()
cat.filter_spatial(region, in_place=True)
Events = np.array([[i, j] for i, j in zip(cat.get_longitudes(),
                                          cat.get_latitudes())])
tstring = floatcsep.utils.timewindow2str(exp.timewindows[0])

raster_fn = 'basemap_3857.tif'
rds = rxr.open_rasterio(raster_fn)
os.makedirs('forecasts', exist_ok=True)

for model in exp.models[:1]:

    model_name = model.name
    model.create_forecast(tstring)
    forecast = model.get_forecast(tstring)
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection=projection)
    rds.plot.imshow(ax=ax, transform=projection, add_labels=False)
    ax = forecast.plot(ax=ax, plot_args=fc_prop)
    ax = cat.plot(ax=ax, plot_args=cat_prop, extent=region.get_bbox())

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Add a title inside the axis with a little offset
    x_title = x_max - 0.02 * (x_max - x_min)  # Adjust the offset here
    y_title = y_max - 0.02 * (y_max - y_min)  # Adjust the offset here
    ax.text(x_title, y_title, model.name, fontsize=21, ha='right', va='top')

    plt.savefig(f'forecasts/{forecast.name}.png', dpi=200, fmt='png',
                facecolor=(0, 0, 0, 0), bbox_inches='tight', pad_inches=0)
    plt.show()


fc_prop.update({'include_cbar': True,
               'cmap': 'magma',
               'alpha_exp': 0.3,
               'clim': [-7, 0],
               'clabel': r'$\log_{10}\lambda_{M\geq4.95}$',
               'clabel_fontsize': 22,
                'cticks_fontsize': 16,
               'projection': projection})

cat_prop.update({'legend_fontsize': 18, 'legend_titlesize': 20,
                 'mag_ticks': [5.0, 5.4, 5.8, 6.1], 'legend': True})

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection=projection)
ax = forecast.plot(ax=ax, plot_args=fc_prop)
cat.plot(ax=ax, plot_args=cat_prop, extent=region.get_bbox())
plt.savefig(f'forecasts/axis_handles.svg', dpi=200, fmt='svg',
            facecolor=(0, 0, 0, 0))
plt.show()