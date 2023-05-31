
import os
# import fiona
import numpy
from floatcsep.experiment import Experiment
# import pickle
import matplotlib.pyplot as plt
import numpy as np
import csep
from csep.core.poisson_evaluations import _simulate_catalog
import itertools
from multiprocessing import Pool

from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import globalenv
import cartopy
import seaborn as sns
import time
import seaborn
import scipy.interpolate as si



def sim(num_events, sampling_weights, sim_fore, random_numbers=None):
    if random_numbers is None:
        random_numbers = numpy.random.rand(num_events)
    else:
        pass
    sim_fore.fill(0)
    pnts = numpy.searchsorted(sampling_weights, random_numbers, side='right')
    numpy.add.at(sim_fore, pnts, 1)
    return sim_fore


def simulate(model, number_events):

    lons, lats = model.region.midpoints().T
    forecast_data = model.spatial_counts()
    sampling_weights = numpy.cumsum(forecast_data) / numpy.sum(forecast_data)
    sim_fore = numpy.zeros(sampling_weights.shape)

    sim_cat = sim(number_events, sampling_weights, sim_fore).astype(int)
    ind_sim = np.where(sim_cat != 0)[0]

    Lons = lons[ind_sim].repeat(sim_cat[ind_sim])
    Lats = lats[ind_sim].repeat(sim_cat[ind_sim])
    noise = np.random.normal(0.0, 0.04, (Lons.shape[0], 2))
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
#
# def pcf_spatstat(fv, method='c'):
#
#     ## Import R spatstat modules
#     spat_core = rpackages.importr("spatstat.core")
#
#     ## Call the pair correlation function
#     pcf_func = spat_core.pcf
#     pcf_args = (('X', fv), ('method', method))
#     pcf_results = pcf_func.rcall(pcf_args, globalenv)
#     return pcf_results


def K_ripley(points, model, r, normpower=2, plot=False):

    ## Import R spatstat modules
    spat_geom = rpackages.importr("spatstat.geom")
    spat_core = rpackages.importr("spatstat.core")

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
    polygon = numpy.genfromtxt('../../catalogs/region_it.txt') # todo fix
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
                ('normpower', normpower))
    else:
        args = (('X', PPP_R), ('lambda', image), ('correction', 'best'),
                ('normpower', normpower), ('r', robjects.FloatVector(r)))

    # Get results
    k_inhomogeneous = spat_core.Kinhom
    k_results = k_inhomogeneous.rcall(args, globalenv)

    if plot:
        spat_geom.plot_im(image, 'asd', multiplot=True)
        spat_geom.plot_ppp(PPP_R, 'asd', add=True)

    return k_results

# def plot_results(Results, alpha=0.05):
#
#
#     L_diff = dict.fromkeys(Results.keys())
#     pcf_diff = dict.fromkeys(Results.keys())
#
#     a = 0
#     for key, value in Results.items():
#         a += 1
#         if a == 6:
#             break
#         K_sims = value['K_sims']
#         K_cat = value['K_cat']
#         r_cat = value['rk_cat']
#         r_sims = value['Rk_sims'][0]
#
#
#
#         K_avg = np.nanmean(K_sims, axis=0)
#         K_up = np.nanquantile(K_sims, 1-alpha/2, axis=0)
#         K_down = np.nanquantile(K_sims, alpha/2, axis=0)
#
#         sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family':'Ubuntu'})
#         # for i in K_sims:
#         #     sns.lineplot(x=r_sims, y=i, lw=0.05, color='black')
#         # sns.lineplot(x=r_cat, y=K_cat, color='r', label='Observed catalog')
#         # sns.lineplot(x=r_sims, y=K_avg, label='Sim. average')
#
#         # plt.fill_between(r_sims, K_down, K_up, color='gray', alpha=0.2, label=r'Sim. envelope ($\alpha=%.2f$)' % (1-2*alpha))
#         # plt.title("Model: %s" % key, fontsize=16)
#         # plt.xlabel(r"$r~~\mathrm{[km]}$")
#         # plt.ylabel(r"$\hat{K}(r)$")
#         # plt.legend(loc=2)
#         # plt.savefig(paths.get_kripley_figpath('K', 10,  key))
#         # plt.show()
#         #
#         L_sims =  value['L_sims']
#         L_cat = value['L_cat']
#         L_avg = np.mean(L_sims, axis=0)
#         L_up = np.quantile(L_sims, 1-alpha/2, axis=0)
#         L_down = np.quantile(L_sims, alpha/2, axis=0)
#
#         L_over = (L_cat > L_up)*(L_cat - L_up) +\
#                  (L_down < L_cat)*(L_cat < L_up)*0 +\
#                  (L_cat < L_down)*(L_cat - L_down)
#         L_over[np.isnan(L_over)] = 0.
#         L_diff[key] = L_over
#
#         # for i in L_sims:
#         #     sns.lineplot(x=r_sims, y=i, lw=0.05, color='black', ls='--')
#         # sns.lineplot(x=r_cat, y=L_cat, color='r', label='Observed catalog')
#         # sns.lineplot(x=r_sims, y=L_avg, label='Sim. average')
#         # plt.fill_between(r_sims, L_down, L_up, color='gray', alpha=0.4, label=r'Sim. envelope ($\gamma=%.2f$)' % (1 - alpha))
#         # # plt.title("Model - %s" % key, fontsize=16)
#         # plt.xlabel(r"$r~~\mathrm{[km]}$")
#         # plt.ylabel(r"$L(r) = \sqrt{\frac{\hat{K}(r)}{\pi}}$")
#         # plt.xlim([None, np.max(r_cat)])
#         # plt.ylim([None, 1000])
#         # plt.legend(loc=2, title=key)
#         # plt.savefig(paths.get_kripley_figpath('L', 10, key))
#         # plt.show()
#
#         R_pcf = r_cat[1:]
#         pcf_sims = []
#         for k in K_sims:
#             # print(key, k[:10])
#             interpolant = si.BSpline(R_pcf, k[1:], k=3).derivative(1)
#             # print(interpolant(R_pcf) / (2*np.pi) / R_pcf)
#             pcf_sims.append(interpolant(R_pcf) / (2*np.pi) / R_pcf)
#
#         fit = si.BSpline(R_pcf, K_cat[1:], k=3).derivative(1)
#         pcf_cat = fit(R_pcf) / 2 / np.pi / R_pcf  # todo
#
#         pcf_avg = np.mean(pcf_sims, axis=0)
#         pcf_up = np.quantile(pcf_sims, 1 - alpha/2, axis=0)
#         pcf_down = np.quantile(pcf_sims, alpha/2, axis=0)
#         # pcf_over = (pcf_cat > pcf_up)*(pcf_cat - pcf_up) +\
#         #            (pcf_down < pcf_cat)*(pcf_cat < pcf_up)*0 +\
#         #            (pcf_cat < pcf_down)*(pcf_cat - pcf_down)
#         # pcf_over[np.isnan(pcf_over)] = 0.
#         # pcf_diff[key] = pcf_over
#         print(len(pcf_sims))
#         for i in pcf_sims:
#             sns.lineplot(x=R_pcf, y=i, lw=0.05, color='black', ls='--')
#         sns.lineplot(x=R_pcf, y=pcf_cat, color='r', label='Observed catalog')
#         g = sns.lineplot(x=R_pcf, y=pcf_avg, label='Sim. average')
#         plt.fill_between(R_pcf, pcf_down, pcf_up, color='gray', alpha=0.2, label=r'Sim. envelope ($\alpha=%.2f$)' % (1-alpha))
#         plt.title("Model - %s" % key, fontsize=16)
#         plt.ylim([-1, None])
#         plt.xlim([-1, 100])
#         plt.xlabel(r"$r~~\mathrm{[km]}$")
#         plt.ylabel(r"$g(r) = \frac{1}{2\pi r}\,\frac{dK(r)}{dr}$")
#         plt.legend(loc=4)
#         plt.savefig(paths.get_kripley_figpath('pcf', 10, key))
#         plt.show()
#         #
#
#     # b = plot_combined(L_diff, pcf_diff, r_sims, pcfr_sims, order=None)
#     # plt.show()
#
#
# def plot_combined(A, B, x, xx, order='inc'):
#     sns.set_style("dark", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
#     from matplotlib import cm
#
#     figsize = (10,5)
#     fig, ax = plt.subplots(figsize=figsize)
#     xlims = []
#     range_ = [np.min(np.array([i for i in A.values()]).ravel()),
#               np.max(np.array([i for i in A.values()]).ravel())]
#
#
#     n = 0
#
#     if order is None:
#         iter_array = A.keys()
#     elif order == 'inc':
#         model_order = np.argsort([np.sum(i != 0) for i in A.values()])
#         iter_array = [list(A.keys())[i] for i in model_order]
#
#     range_ = [-np.max(np.array([i for i in A.values()])), np.max(np.array([i for i in A.values()]))]
#     for index, key in enumerate(iter_array):
#         vals = A[key]
#         vals[np.isnan(vals)] = 0
#         xamp = 1.5
#         new_vals = (vals)/(range_[1])*xamp - 0.5
#         # ax.fill([index]*len(new_vals), new_vals)
#
#         xmin, xmax = index - xamp, index + xamp
#         ymin, ymax = min(x), max(x)
#
#         y = index -0.5 -  new_vals
#
#         img_data = np.flip(np.linspace(range_[0], range_[1], 100))
#         img_data = img_data.reshape(img_data.size, 1).T
#         im = ax.imshow(img_data, aspect='auto', origin='lower', cmap=plt.cm.coolwarm,
#                        extent=[xmin, xmax, ymin, ymax], zorder=n, vmin=-250, vmax=250)
#         b = ax.fill(np.append(y, index), np.append(x, x[-1]), facecolor='none', edgecolor='none', zorder=n)
#         for i in b:
#             im.set_clip_path(i)
#         n += 1
#         ax.fill_betweenx(x, index - 0.5 - new_vals, [index]*len(vals),
#                              facecolor='none', edgecolor='gray', alpha=0.8,
#                              where = (vals!=0), interpolate=True, zorder=n)
#
#         plt.axvline(index, c='black',zorder=n, lw=0.9)
#     plt.grid()
#
#     cba = plt.colorbar(im, shrink=0.5, fraction=0.03, pad=0.03)
#     cba.ax.set_title('$L(r)$', pad=15)
#         # plot and clip the imag
#
#
#
#
#
#
#     n = 0
#     range_ = [0,
#               np.sqrt(np.abs(np.max(np.array([i for i in B.values()]).ravel())))]
#     for index, key in enumerate(iter_array):
#
#         vals = B[key]
#         vals[np.isnan(vals)] = 0
#
#         vals = np.sqrt(np.abs(vals))
#         vals[np.isnan(vals)] = 0
#         vals[-1] = 0
#         xamp = 1.2
#         new_vals = (vals) / (range_[1]) * xamp - 0.5
#
#         xmin, xmax = index - xamp, index
#         ymin, ymax = min(xx), max(xx)
#
#
#         y = index - 0.5 - new_vals
#         img_data = np.linspace( range_[1], range_[0], 100)
#         img_data = np.flip(np.linspace(range_[0], range_[1], num=100))
#         img_data = img_data.reshape(img_data.size, 1).T
#         # plot and clip the imag
#         im = ax.imshow(img_data, aspect='auto', origin='lower', cmap=plt.cm.Greens,
#                        extent=[xmin, xmax, ymin, ymax], alpha=0.9, zorder=n, vmax=25)
#         b = ax.fill(y, xx, facecolor='none', edgecolor='none', zorder=n)
#         for i in b:
#             im.set_clip_path(i)
#         n += 1
#         a = ax.fill_betweenx(xx, y, [index] * len(vals),
#                              facecolor='none', edgecolor='gray', alpha=0.8,
#                              where=(vals != 0), interpolate=True, zorder=n)
#
#     cba = fig.colorbar(im,  shrink=0.5, fraction=0.03, pad=0.03)
#     cba.ax.set_title('$\sqrt{g(r)}$', pad=15)
#
#
#     ax.set_xlim([-1, 19])
#     ax.set_ylim([-10, 800])
#
#     # plt.gca().invert_yaxis()
#
#     ax.set_xticklabels([key for key in iter_array], rotation=45, ha='right')
#     ax.set_xticks(numpy.arange(len(A)))
#     # ax.xaxis.tick_top()
#     ax.set_ylabel("$r\,[\mathrm{km}]$")
#     # plt.title("Ripley's L and Pair Correlation Functions")
#     plt.tight_layout()
#     plt.savefig(paths.get_kripley_figpath('Total', 10, format='png'), dpi=300)
#     plt.show()
#
#     return b
#


def run(nsim=10, nproc=16):
    timewindow = '2010-01-01_2020-01-01'
    cfg_file = os.path.join(os.path.dirname(__file__), '../../runs/total', 'config.yml')
    Exp = Experiment.from_yml(cfg_file)
    Exp.stage_models()
    models = [i.get_forecast(timewindow) for i in Exp.models]
    catalog = np.stack((Exp.catalog.get_longitudes(), Exp.catalog.get_latitudes())).T
    n_events = catalog.shape[0]

    results = dict.fromkeys([i.name for i in models])

    r_disc = 10
    for model in models[:1]:
        r = np.linspace(0, 1, r_disc)**2*800
        k_eval_cat = K_ripley(catalog, model, r=r)
        r_cat = list(k_eval_cat[0])
        k_cat = list(k_eval_cat[2])

        sim_catalogs = [(simulate(model, n_events), model, r) for i in range(nsim)]

        r_sims = []
        k_sims = []
        print('Processing model %s' % model.name)
        start = time.process_time()
        p = Pool(nproc)
        Starmap = p.starmap(K_ripley, sim_catalogs)
        p.close()
        for K in Starmap:
            rk_i = list(K[0])
            K_i = list(K[2])
            r_sims.append(rk_i)
            k_sims.append(K_i)

        print(time.process_time() - start)
        assert numpy.allclose(np.mean(r_sims, axis=0), np.array(r_sims[0]))
        results[model.name] = {'k_sims': k_sims,
                               'r_sims': r_sims,
                               'k_cat': k_cat,
                               'r_cat': r_cat}
    #
    # with open(paths.get_kripley_result_path('K_%s' % nsim, 10), 'wb') as file_:
    #     pickle.dump(Results, file_)

    return results
#

if __name__ == "__main__":
    a = run()
#     Results = run(nsim=100, nproc=12)
#     Results = run(nsim=500, nproc=12)
#     Results = run(nsim=1000, nproc=12)
#     Results = run(nsim=2000, nproc=12)

    # with open(paths.get_kripley_result_path('K', 10), 'rb') as file_:
    #     Results = pickle.load(file_)
    #
    # a = plot_results(Results)

