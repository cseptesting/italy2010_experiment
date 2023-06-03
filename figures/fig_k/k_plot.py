import os
import h5py
import numpy as np
import time
import scipy.interpolate as scint
from multiprocessing import Pool
import cartopy
from floatcsep.experiment import Experiment
from floatcsep.utils import timewindow2str
import seaborn as sns
import matplotlib.pyplot as plt


def read_hdf5(fname):

    with h5py.File(fname, 'r') as db:
        parsed_results = dict.fromkeys(db.keys())
        for key, val in db.items():
            if key != 'name':
                parsed_results[key] = val[:]
            else:
                parsed_results[key] = val

    return parsed_results


def plot_results(result, savefolder='figs', alpha=0.05, show=False,
                 figsize=(6,4)):

    os.makedirs(savefolder, exist_ok=True)
    name = result.get('name', 'None')
    r = result['r']
    k_sims = result['k_sims']
    l_sims = result['l_sims']
    pcf_sims = result['pcf_sims']

    k_cat = result['k_cat']
    l_cat = result['l_cat']
    pcf_cat = result['pcf_cat']

    k_mean = result['k_mean']
    k_upper = result[f'k_{1-alpha/2:.2f}']
    k_lower = result[f'k_{alpha/2:.2f}']

    L_avg = result['l_mean']
    L_up = result[f'l_{1-alpha/2:.2f}']
    L_down = result[f'l_{alpha/2:.2f}']
    L_over = result[f'l_over_{alpha:.2f}']

    pcf_avg = result['pcf_mean']
    pcf_up = result[f'pcf_{1-alpha/2:.2f}']
    pcf_down = result[f'pcf_{alpha/2:.2f}']
    pcf_over = result[f'pcf_over_{alpha:.2f}']


    plt.figure(figsize=figsize)
    sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family':'Ubuntu'})
    for i in k_sims:
        sns.lineplot(x=r, y=i, lw=0.05, color='black')
    sns.lineplot(x=r, y=k_cat, color='r', label='Observed catalog')
    sns.lineplot(x=r, y=k_mean, label='Sim. average')

    plt.fill_between(r, k_lower, k_upper, color='gray', alpha=0.2,
                     label=r'Sim. envelope ($\alpha=%.2f$)' % (1-alpha))
    plt.title("Model: %s" % name, fontsize=16)
    plt.xlabel(r"$r~~\mathrm{[km]}$")
    plt.ylabel(r"$\hat{K}(r)$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, f'K_{name}.png'), dpi=200,
                             facecolor=(0, 0, 0, 0))
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    for i in l_sims:
        sns.lineplot(x=r, y=i, lw=0.05, color='black', ls='--')
    sns.lineplot(x=r, y=l_cat, color='r', label='Observed catalog')
    sns.lineplot(x=r, y=L_avg, label='Sim. average')
    plt.fill_between(r, L_down, L_up, color='gray', alpha=0.4,
                     label=r'Sim. envelope ($\gamma=%.2f$)' % (1 - alpha))
    plt.title("Model - %s" % name, fontsize=16)
    plt.xlabel(r"$r~~\mathrm{[km]}$")
    plt.ylabel(r"$L(r) = \sqrt{\frac{\hat{K}(r)}{\pi}}$")
    plt.xlim([None, np.max(r)])
    plt.ylim([None, 1000])
    plt.legend(loc=2)
    plt.savefig(os.path.join(savefolder, f'L_{name}.png'), dpi=200,
                facecolor=(0, 0, 0, 0))
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    for i in pcf_sims:
        sns.lineplot(x=r, y=i, lw=0.05, color='black', ls='--')
    sns.lineplot(x=r, y=pcf_cat, color='r', label='Observed catalog')
    g = sns.lineplot(x=r, y=pcf_avg, label='Sim. average')
    plt.fill_between(r, pcf_down, pcf_up, color='gray', alpha=0.2, label=r'Sim. envelope ($\alpha=%.2f$)' % (1-alpha))
    plt.title("Model - %s" % name, fontsize=16)
    plt.ylim([-1, None])
    plt.xlim([-1, 100])
    plt.xlabel(r"$r~~\mathrm{[km]}$")
    plt.ylabel(r"$g(r) = \frac{1}{2\pi r}\,\frac{dK(r)}{dr}$")
    plt.legend(loc=1)
    plt.savefig(os.path.join(savefolder, f'pcf_{name}.png'), dpi=200,
                facecolor=(0, 0, 0, 0))

    if show:
        plt.show()

    L_offset = np.insert(L_over, -1, 0)
    pcf_offset = np.insert(pcf_over, [0, -1], [pcf_over[0], 0])

    return L_offset, pcf_offset, r


def plot_combined(total_results, savefolder='figs', alpha=0.05, order='inc'):

    tag_a = f'l_over_{alpha:.2f}'
    tag_b = f'pcf_over_{alpha:.2f}'
    A = {i: j[tag_a] for i, j in total_results.items()}
    B = {i: j[tag_b] for i, j in total_results.items()}
    x = list(total_results.values())[0]['r']
    figsize = (10, 5)
    fig, ax = plt.subplots(figsize=figsize)
    n = 0
    if order is None:
        iter_array = A.keys()
    elif order == 'inc':
        model_order = np.argsort([np.sum(i != 0) for i in A.values()])
        iter_array = [list(A.keys())[i] for i in model_order]
    else:
        iter_array = order
    range_ = [-np.max(np.array([i for i in A.values()])),
              np.max(np.array([i for i in A.values()]))]
    for index, key in enumerate(iter_array):

        vals = A[key]
        vals[np.isnan(vals)] = 0
        xamp = 1.5
        new_vals = vals/(range_[1])*xamp - 0.5
        xmin, xmax = index - xamp, index + xamp
        ymin, ymax = min(x), max(x)

        y = index - 0.5 - new_vals

        img_data = np.flip(np.linspace(range_[0], range_[1], 100))
        img_data = img_data.reshape(img_data.size, 1).T
        im = ax.imshow(img_data, aspect='auto', origin='lower',
                       cmap=plt.cm.coolwarm, extent=[xmin, xmax, ymin, ymax],
                       zorder=n, vmin=-250, vmax=250)
        b = ax.fill(np.append(y, index), np.append(x, x[-1]),
                    facecolor='none', edgecolor='none', zorder=n)

        for i in b:
            im.set_clip_path(i)
        n += 1
        ax.fill_betweenx(x, index - 0.5 - new_vals, [index]*len(vals),
                             facecolor='none', edgecolor='gray', alpha=0.8,
                             where=(vals != 0), interpolate=True, zorder=n)

        plt.axvline(index, c='black', zorder=n, lw=0.9)
    plt.grid()
    cba = plt.colorbar(im, shrink=0.5, fraction=0.03, pad=0.03)
    cba.ax.set_title('$L(r)$', pad=15)
    n = 0
    range_ = [0, np.sqrt(np.abs(np.max(np.array(
        [i for i in B.values()]).ravel())))]
    for index, key in enumerate(iter_array):

        vals = B[key]
        vals[np.isnan(vals)] = 0

        vals = np.sqrt(np.abs(vals))
        vals[np.isnan(vals)] = 0
        vals[-1] = 0
        xamp = 1.2
        new_vals = vals / (range_[1]) * xamp - 0.5

        xmin, xmax = index - xamp, index
        ymin, ymax = min(x), max(x)

        y = index - 0.5 - new_vals
        img_data = np.flip(np.linspace(range_[0], range_[1], num=100))
        img_data = img_data.reshape(img_data.size, 1).T
        im = ax.imshow(img_data, aspect='auto', origin='lower',
                       cmap=plt.cm.Greens, extent=[xmin, xmax, ymin, ymax],
                       alpha=0.9, zorder=n, vmax=25)
        b = ax.fill(y, x, facecolor='none', edgecolor='none', zorder=n)
        for i in b:
            im.set_clip_path(i)
        n += 1
        ax.fill_betweenx(x, y, [index] * len(vals), facecolor='none',
                         edgecolor='gray', alpha=0.8, where=(vals != 0),
                         interpolate=True, zorder=n)

    cba = fig.colorbar(im,  shrink=0.5, fraction=0.03, pad=0.03)
    cba.ax.set_title('$\sqrt{g(r)}$', pad=15)
    ax.set_xlim([-1, 19])
    ax.set_ylim([-10, 800])
    ax.set_xticklabels([key for key in iter_array], rotation=45, ha='right')
    ax.set_xticks(np.arange(len(A)))
    ax.set_ylabel("$r\,[\mathrm{km}]$")
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, 'total_k.png'), dpi=300,
                facecolor=(0, 0, 0, 0))
    plt.show()

    return b


if __name__ == "__main__":
    exp_path = os.path.join('../../src', 'total')
    ripley_path = os.path.join('../../src', 'ripley')
    cfg_file = os.path.join(exp_path, 'config.yml')
    experiment = Experiment.from_yml(cfg_file)
    experiment.stage_models()
    experiment.set_tasks()
    time_window = timewindow2str(experiment.timewindows[0])
    models = experiment.models
    results = {}

    l_offset = dict.fromkeys([i.name for i in models])
    pcf_offset = dict.fromkeys([i.name for i in models])

    for i, model in enumerate(models):
        results[model.name] = read_hdf5(os.path.join(ripley_path,
                                                     'results',
                                                     f'K_{model.name}.hdf5'))
        results[model.name]['name'] = model.name
        plot_results(results[model.name])

    order = [(i.sim_name, i.observed_statistic) for i in
             experiment.read_results(experiment.tests[-4], time_window)]
    order = [order[i][0] for i in np.flip(np.argsort([i[1] for i in order]))]

    plot_combined(results, order=order)


