import numpy
import seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.lines import Line2D
from datetime import datetime

import csep
from floatcsep import experiment
from floatcsep.utils import plot_sequential_likelihood
seaborn.set_style("white", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
plt.rcParams.update({
    'axes.titleweight': 'bold',
    'xtick.bottom': True, 'axes.labelweight': 'bold',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 9})


def plot_seq_ig(experiment, figpath=None):

    catalog = experiment.catalog
    ig_seq = [i for i in experiment.tests if i.name == 'Sequential_IG'][0]
    ig_results = experiment.read_results(ig_seq, experiment.timewindows[-1])
    rank = numpy.argsort([i.observed_statistic[-1] for i in ig_results])[::-1]
    ig_results = [ig_results[i] for i in rank]
    colors = ['steelblue', 'darkturquoise',
              'darkgreen', 'limegreen',
              'xkcd:off yellow', 'gold', 'xkcd:yellowish orange',
              'xkcd:raw umber', 'xkcd:brown', 'xkcd:cement',
              'xkcd:fuchsia', 'xkcd:pinkish purple', 'xkcd:purple',
              'xkcd:indigo', 'xkcd:shocking pink',
              'xkcd:medium pink', 'red', 'xkcd:darkish red', 'xkcd:black']
    linestyles = ['-', '-',  '-', '-', '-',  '-', '-', '-',  '-', '-',  '-',
                  ':', ':', ':',  '-.', '-.', '--', '--', '--']
    startyear = experiment.timewindows[0][0]
    endyears = [j[1] for j in experiment.timewindows]
    years = [startyear] + endyears
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for i, result in enumerate(ig_results):
        data = [0] + result.observed_statistic
        ax.plot(years, data, color=colors[i],
                linewidth=1.2, linestyle=linestyles[i],
                marker='o',
                markersize=2,
                label=result.sim_name)
        ax.set_ylabel('Information Gain\n(respect to MPS04)')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_xlim([startyear, None])
        ax.grid(True)
    ax.legend(loc=(1.07, 0), fontsize=7)
    ax.set_ylim([-20, 16])

    plt.show()

    ax2 = ax.twinx()
    mag = catalog.data['magnitude']
    time = [datetime.fromtimestamp(i / 1000.) for i in
            catalog.data['origin_time']]
    ax2.scatter(time, mag, marker='o', s=2**((mag-4) * 4),
                edgecolor='gray', color='r', alpha=0.2)
    ax2.set_ylim([3., 6.2])
    ax2.set_ylabel('$M_w$', fontsize=14, loc='top', rotation='horizontal')
    ax2.set_yticks([5., 5.5, 6.0])
    ax2.yaxis.set_label_coords(1.06, 1.1)
    fig.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=300)
    plt.show()


if __name__ == '__main__':
    cfg = '../../src/sequential/config.yml'
    experiment = experiment.Experiment.from_yml(cfg)
    experiment.set_tasks()
    cl_figpath = 'Sequential_IG.png'
    plot_seq_ig(experiment, cl_figpath)
