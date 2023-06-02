import os
import h5py
import numpy as np
import time
import scipy.interpolate as scint
from multiprocessing import Pool
import cartopy
from floatcsep.experiment import Experiment
from floatcsep.utils import timewindow2str
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import globalenv


def sim(num_events, sampling_weights, sim_fore, random_numbers=None):
    if random_numbers is None:
        random_numbers = np.random.rand(num_events)
    else:
        pass
    sim_fore.fill(0)
    pnts = np.searchsorted(sampling_weights, random_numbers, side='right')
    np.add.at(sim_fore, pnts, 1)
    return sim_fore


def simulate(model, number_events):

    lons, lats = model.region.midpoints().T
    forecast_data = model.spatial_counts()
    sampling_weights = np.cumsum(forecast_data) / np.sum(forecast_data)
    sim_fore = np.zeros(sampling_weights.shape)

    sim_cat = sim(number_events, sampling_weights, sim_fore).astype(int)
    ind_sim = np.where(sim_cat != 0)[0]

    Lons = lons[ind_sim].repeat(sim_cat[ind_sim])
    Lats = lats[ind_sim].repeat(sim_cat[ind_sim])
    noise = np.random.uniform(-0.04999, 0.04999, (Lons.shape[0], 2))
    events = np.array([Lons, Lats]).T + noise

    return events


def lonlat2mercator(points):
    """
    Defined to system RDN2008 - Italy (epsg:6875)
    """
    src_crs = cartopy.crs.Geodetic()
    dest_crs = cartopy.crs.Mercator()
    Points = dest_crs.transform_points(src_crs, points[:, 0], points[:, 1])
    return np.floor(Points[:, :2])


def ripley2hdf5(results, fname, grp='',):

    with h5py.File(fname, 'w') as hfile:
        for i, j in results.items():
            if isinstance(j, (list, np.ndarray)):
                array = np.array(j)
                hfile.require_dataset(f'{grp}/{i}', shape=array.shape, dtype=float)
                hfile[f'{grp}/{i}'][:] = array
            if isinstance(j, str):

                hfile.attrs[f'{i}'] = j


def read_hdf5(fname):

    with h5py.File(fname, 'r') as db:
        parsed_results = dict.fromkeys(db.keys())
        for key, val in db.items():
            parsed_results[key] = val[:]
    return parsed_results


def k_ripley(points, model, polygon, r, norm_power=2, k_fit=2, plot=False):

    ## Import R spatstat modules
    spat_geom = rpackages.importr("spatstat.geom")
    spat_core = rpackages.importr("spatstat.core")
    # spat_core = rpackages.importr("spatstat")

    # Import model to R
    region = model.region
    # Get the region properties
    midpoints = region.midpoints()
    lon_r = np.sort(np.unique(midpoints[:, 0]))
    lat_r = np.sort(np.unique(midpoints[:, 1]))
    X_r = lonlat2mercator(np.stack([lon_r, np.repeat(min(lat_r), lon_r.shape[0])]).T)[:,0]/1000
    Y_r = lonlat2mercator(np.stack([np.repeat(min(lon_r), lat_r.shape[0]), lat_r]).T)[:,1]/1000
    Xregion_array = robjects.FloatVector(np.sort(np.unique(X_r)))
    Yregion_array = robjects.FloatVector(np.sort(np.unique(Y_r)))

    # Get the model rates
    rates = model.spatial_counts(cartesian=True)
    rates_array = robjects.FloatVector(rates.T.ravel())
    rates_mat = robjects.r['matrix'](rates_array, nrow=rates.shape[0])
    image = spat_geom.im(rates_mat, Xregion_array, Yregion_array)

    # Get the polygon window of the forecast
    polygon = np.genfromtxt(polygon)
    polygon = lonlat2mercator(polygon)/1000.
    poly_array = robjects.FloatVector(np.flipud(polygon[:-1, :]).T.ravel())
    poly_mat = robjects.r['matrix'](poly_array, nrow=polygon.shape[0] - 1)
    poly_array_xrange = robjects.FloatVector([np.min(polygon[:, 0]), np.max(polygon[:, 0])])
    poly_array_yrange = robjects.FloatVector([np.min(polygon[:, 1]), np.max(polygon[:, 1])])
    window = spat_geom.owin(xrange=poly_array_xrange, yrange=poly_array_yrange, poly=poly_mat)

    # Convert the catalog into R point process model
    Points = lonlat2mercator(points)/1000

    point_array_x = robjects.FloatVector(Points[:, 0])
    point_array_y = robjects.FloatVector(Points[:, 1])
    PPP_R = spat_geom.ppp(point_array_x, point_array_y, window)

    # Get arguments: r-vector user-defined or automatic
    if r is None:
        args = (('X', PPP_R), ('lambda', image), ('correction', 'best'),
                ('normpower', norm_power))
    else:
        args = (('X', PPP_R), ('lambda', image), ('correction', 'best'),
                ('normpower', norm_power), ('r', robjects.FloatVector(r)))

    # Get results
    k_inhomogeneous = spat_core.Kinhom
    k_results = k_inhomogeneous.rcall(args, globalenv)

    if plot:
        spat_geom.plot_im(image, 'asd', multiplot=True)
        spat_geom.plot_ppp(PPP_R, 'asd', add=True)

    r_i = np.array(k_results[0])
    k_i = np.array(k_results[2])
    l_i = np.divide(np.sqrt(k_i), np.sqrt(np.pi))

    # low_cutoff
    cell_size = 5
    degree = 3
    ind = np.argwhere(r_i > cell_size)[0, 0]
    r_pcf = r_i[ind:-degree]

    interpolation = scint.BSpline(r_i, k_i, k=k_fit).derivative(1)
    pcf_i_ = interpolation(r_pcf) / (2 * np.pi) / r_pcf
    pcf_i = np.zeros(len(r_i))
    pcf_i[ind:-degree] = pcf_i_

    return k_i, l_i, pcf_i


def k_ripley_test(model, catalog, polygon='region_it.txt', nsim=100, r_disc=100,
                  alpha=0.05):

    print('Processing model %s' % model.name)
    catalog = np.stack((catalog.get_longitudes(), catalog.get_latitudes())).T
    n_events = catalog.shape[0]

    r = np.linspace(0, 1, r_disc)**2*800
    k_eval_cat = k_ripley(catalog, model, polygon, k_fit=1, r=r)
    k_cat = np.array(k_eval_cat[0])
    l_cat = np.array(k_eval_cat[1])
    pcf_cat = np.array(k_eval_cat[2])

    sim_catalogs = [(simulate(model, n_events), model, polygon, r)
                    for i in np.arange(nsim)]

    start = time.perf_counter()
    p = Pool(processes=os.cpu_count())
    starmap = p.starmap(k_ripley, sim_catalogs)
    p.close()

    k_sims = np.array([row[0] for row in starmap])
    l_sims = np.array([row[1] for row in starmap])
    pcf_sims = np.array([row[2] for row in starmap])

    k_mean = np.nanmean(k_sims, axis=0)
    k_upper = np.nanquantile(k_sims, 1-alpha/2, axis=0)
    k_lower = np.nanquantile(k_sims, alpha/2, axis=0)

    l_mean = np.mean(l_sims, axis=0)
    l_upper = np.quantile(l_sims, 1-alpha/2, axis=0)
    l_lower = np.quantile(l_sims, alpha/2, axis=0)
    l_over = (l_cat > l_upper)*(l_cat - l_upper) + \
             (l_lower < l_cat)*(l_cat < l_upper)*0 + \
             (l_cat < l_lower)*(l_cat - l_lower)
    l_over[np.isnan(l_over)] = 0.

    pcf_mean = np.mean(pcf_sims, axis=0)
    pcf_upper = np.quantile(pcf_sims, 1 - alpha/2, axis=0)
    pcf_lower = np.quantile(pcf_sims, alpha/2, axis=0)
    pcf_over = (pcf_cat > pcf_upper) * (pcf_cat - pcf_upper) + \
               (pcf_lower < pcf_cat) * (pcf_cat < pcf_upper) * 0 + \
               (pcf_cat < pcf_lower) * (pcf_cat - pcf_lower)
    pcf_over[np.isnan(pcf_over)] = 0.

    results = {'name': model.name,
               'r': r,

               'k_sims': k_sims,
               'k_mean': k_mean,
               f'k_{1-alpha/2:.2f}': k_upper,
               f'k_{alpha/2:.2f}': k_lower,

               'l_sims': l_sims,
               'l_mean': l_mean,
               f'l_{1-alpha/2:.2f}': l_upper,
               f'l_{alpha/2:.2f}': l_lower,
               f'l_over_{alpha:.2f}': l_over,

               'pcf_sims': pcf_sims,
               'pcf_mean': pcf_mean,
               f'pcf_{1-alpha/2:.2f}': pcf_upper,
               f'pcf_{alpha/2:.2f}': pcf_lower,
               f'pcf_over_{alpha:.2f}': pcf_over,

               'k_cat': k_cat,
               'l_cat': l_cat,
               'pcf_cat': pcf_cat}
    print(f'Finished in {time.perf_counter() - start:.1f}s')

    return results


