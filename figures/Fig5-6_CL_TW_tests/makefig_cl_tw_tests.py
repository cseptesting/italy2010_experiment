import shutil
import numpy
import seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.lines import Line2D

import csep
from floatcsep import experiment
seaborn.set_style("white", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
plt.rcParams.update({'xtick.bottom': True, 'axes.labelweight': 'bold',
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 9})


def plot_cl_test(experiment_obj, figpath=None):

    cl_test = [i for i in experiment_obj.tests if i.name == 'Poisson_CL'][0]
    cl_results = experiment_obj.read_results(cl_test, experiment_obj.timewindows[-1])
    rank = numpy.argsort([i.quantile for i in cl_results])[::-1]
    cl_results = [cl_results[i] for i in rank]
    ax = csep.utils.plots.plot_poisson_consistency_test(
                                cl_results,
                                plot_args=cl_test.plot_args[0],
                                **cl_test.plot_kwargs[0])
    for i, j in enumerate(cl_results):
        ax.plot(numpy.mean(j.test_distribution),
                len(cl_results) - i - 1, 'ko', markersize=4)
        ax.xaxis.set_major_locator(tick.MaxNLocator(integer=True))

    legend_elements = [Line2D([0], [0], marker='o', lw=0, color='k',
                              label='Sim. expected value', markersize=2),
                       Line2D([0], [0], color='k', lw=1,
                              label='Sim. 95% conf.'),
                       Line2D([0], [1], color='green', marker='s', lw=0,
                              markersize=4, label='Log-likelihood (passes)'),
                       Line2D([0], [1], color='red', marker='o', lw=0,
                              markersize=4, label='Log-likelihood (fails)')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax2 = ax.twinx()
    ax2.set_yticks(numpy.arange(len(cl_results)))
    ax2.set_yticklabels(
        [f'{res.quantile * 100:.1f}' for res in cl_results[::-1]])
    ax2.set_ylim([-0.5, len(cl_results) - 0.5])
    ax.get_figure().text(0.91, 0.95, r'$\gamma^{\,j}\,[\%]$')
    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=1500)
    plt.show()


def plot_tw_test(experiment_obj, figpath=None):

    fig0 = experiment_obj.path(experiment_obj.timewindows[0],
                               'figures',
                               'Matrix_T')
    shutil.copy(fig0 + '.png', figpath)


if __name__ == '__main__':
    cfg = '../../src/total/config.yml'
    exp = experiment.Experiment.from_yml(cfg)
    exp.set_tasks()

    cl_figpath = 'Fig5_HR.png'
    tw_figpath = 'Fig6_HR.png'
    plot_cl_test(exp, cl_figpath)
    plot_tw_test(exp, tw_figpath)
