import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn
from fecsep import experiment
import json
import csep
from csep.models import EvaluationResult
seaborn.set_style("white", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
plt.rcParams.update({'xtick.bottom': True, 'axes.labelweight': 'bold',
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 9})

def norm(array):
    return (array - array.min())/(array.max() -array.min())

def plot_consistency(ax, tests, model_i, color):
    Y_offset = 1.2 * 1.24
    pos = ax.get_figure().transFigure.inverted().transform(
        ax.transData.transform([(i, Y_offset ) for i in ax.get_xticks()]))[model_i]
    r = 0.01
    sep =1.2
    dh = -0.02
    cx = [pos[0] - sep*r*(len(tests)-1) + 2*sep*r*(i) for i in range(len(tests))]
    for i, ci in zip(tests, cx):
        artist = patches.Circle((ci, pos[1] + dh), r, fc=color, ec='black')
        ax.get_figure().add_artist(artist)
        ax.get_figure().text(ci, pos[1] + dh, i, ha='center', va='center', color='white')
    return ax

def plot_all_consistencies(Axes, Results, color='green'):
    for i, test_pass in enumerate(Results):
        plot_consistency(Axes, test_pass, i, color=color)
    return Axes

def plot_axis(axis, n_results, offset, end_theta, n, min_y,
              array, yticks, low_bound, color, label, fontsize, format,
              ytick_offset=0.03):

    axis_angle = (end_theta + np.deg2rad(offset) + n*(2*np.pi-end_theta - 2*np.deg2rad(offset))/(n_results-1))
    axis.plot([axis_angle, axis_angle], [min_y, 1. + min_y],
              color=color, linewidth=2, linestyle=':', alpha=0.5)


    exp = None
    for i in yticks:
        val = (i - min_y) * (array.max() - array.min()) + array.min()
        if 'e' in format:
            val = format % val
            # val, exp = sig_exp(str(val))
            val, exp = str(val).split('e')

        else:
            val = format % val
        if low_bound and i == yticks[0]:
            val = '<' + val

        axis.text(axis_angle, i - ytick_offset, val,
                  rotation=np.rad2deg(axis_angle) + 90, color=color,
                  ha='center', va='center', fontsize=fontsize)
    if 'e' in format:
        label += '\n ($\cdot\,10^{{%i}}$)' % int(exp)

    axis.text(axis_angle, i + 0.05, '%s' % label,
              color=color, ha='left', va='center', fontsize=10)
    if exp:
        return exp

    # axis.text(axis_angle, i + 0.15, label, color=color,
    #         ha='center', va='center', fontsize=14)

def plot_theta_seps(axis, name_models, end_angle, n_model, n_results, width_model,min_y=0):
    for i in np.linspace(0.0, end_angle, n_model + 1):  # ygrid separators
        axis.plot([i , i],
                  [min_y, 1. + min_y], color='grey', linewidth=0.5, zorder=3)
    axis.get_xaxis().set_ticks(np.linspace(0.0, end_angle , n_model,
                                         endpoint=False) + n_results/2*width_model)
    axis.get_xaxis().set_ticklabels(name_models)
    axis.tick_params(axis='x', which='major', pad=20)

def plot_rticks(axis, min_r, n_r):
    axis.get_yaxis().grid(zorder=0, label=False)
    yticks = [min_r]
    yticks.extend([i + min_r for i in np.linspace(0,1,n_r+1)])
    axis.get_yaxis().set_ticks(yticks)
    axis.get_yaxis().set_ticklabels([])
    axis.set_ylim([0, 1.0 + min_r])
    return yticks

def plot_scores(arrays, colors, result_labels, model_labels,
                lowbounds, format, angle_offset=90, offset=10, min_y=0.3, ny=4, fontsize=9):

    # N plots
    n_results = len(arrays)
    n_models = len(arrays[0])

    # Plot properties
    figsize = (8,8)
    end_theta = 2 * np.pi - angle_offset / 360 * 2 * np.pi
    width = end_theta/n_models/n_results
    theta = np.linspace(0.0, end_theta, n_results * n_models, endpoint=False)

    # Rearange results and colors

    tuple_arrays = tuple([norm(i) for i in arrays])
    score_total = np.vstack(tuple_arrays).T.ravel()

    color_array = [plt.cm.colors.to_rgb(i) for i in colors]
    tuple_colors = tuple([[color for i in range(n_models)] for color in color_array])
    colors_full = np.hstack(tuple_colors).reshape((n_models*n_results,3))

    # Create figure
    ax = plt.subplot(111,projection='polar')
    fig = ax.get_figure()
    fig.set_size_inches(figsize)
    ax.grid(False)

    # Plot Data

    ax.bar(theta + width/2., score_total, width=width, bottom=min_y,
           color=colors_full, alpha=1, zorder=0)

    # Plot shaded region
    ax.bar(np.pi+end_theta/2, 1 + min_y, width= 2*np.pi - end_theta, color='grey', alpha=0.2, bottom=min_y)
    ax.bar(0, min_y, width= 2*np.pi , color='grey', alpha=0.2)

    # Plot auxiliaries
    plot_theta_seps(ax, model_labels, end_theta, n_models,n_results, width, min_y=min_y)
    yticks = plot_rticks(ax, min_y, ny)

    # todo make generic
    new_labels = ['Log-score', ' Binomial-score', 'Brier-score']
    for k, i in enumerate([0, 1, 2]):
        if k == 0 or k == 2:
            ytick_offset = 0.03
        else:
            ytick_offset = -0.03
        plot_axis(ax, 3, offset, end_theta, k,  min_y, arrays[i], yticks, lowbounds[i],
                  color_array[i], new_labels[i], fontsize, format[i], ytick_offset)

    return ax

def plot_legends(Axes, colors, labels):
    legend_elements = [Line2D([0], [0], color=plt.cm.colors.to_rgb(colors[0]),
                              lw=4, label=labels[0]),
                       Line2D([0], [0], color=plt.cm.colors.to_rgb(colors[1]),
                              lw=4, label=labels[1]),
                       Line2D([0], [0], color=plt.cm.colors.to_rgb(colors[2]),
                              lw=4, label=labels[2]),
                       Line2D([0], [0], color=colors[3],
                              lw=0, marker='o', markeredgecolor='black', markersize=10, label=labels[3])]

    Axes.get_figure().legend(handles=legend_elements, loc='lower right', fontsize=9)
    return Axes

def plot_results(exp, labels, p=0.01, lowcuts=False, show=True,
                 format='%.2f', savepath=None):


    if isinstance(format, str):
        format = [format] * len(labels)
    if lowcuts is False:
        lowcuts = [lowcuts] * len(labels)

    ## Log-Likelihood
    models = [i.name for i in exp.models]

    ll_test = [i for i in exp.tests if i.name == 'Poisson_CL'][0]
    # _, _ , paths = exp.prepare_paths()
    LL = exp._read_results(ll_test, exp.timewindows[-1])
    # LL = []
    # for model in models:
    #     with open(paths['evaluations']['Poisson_L'][model], 'r') as file_:
    #         LL.append(EvaluationResult.from_dict(json.load(file_)))
    ll_score = np.array([i.observed_statistic for i in LL])
    ll_label = r'$\mathcal{L}$'

    if isinstance(lowcuts[0], (float, int)):
        ll_score[ll_score < lowcuts[0]] = lowcuts[0]


    ## Binomial_S
    # BS = []
    bs_test = [i for i in exp.tests if i.name == 'Binomial_S'][0]
# _, _ , paths = exp.prepare_paths()
    BS = exp._read_results(bs_test, exp.timewindows[-1])
    # for model in models:
    #     with open(paths['evaluations']['Binomial_S'][model], 'r') as file_:
    #         BS.append(EvaluationResult.from_dict(json.load(file_)))
    bs_score = np.array([i.observed_statistic for i in BS])

    bs_label = r'$\mathcal{S}_{B}$'
    if isinstance(lowcuts[1], (float, int)):
        bs_score[bs_score < lowcuts[1]] = lowcuts[1]

    ## Brier score
    Brier_results = []
    Brier_test = [i for i in exp.tests if i.name == 'Brier'][0]
# _, _ , paths = exp.prepare_paths()
    Brier_results = exp._read_results(Brier_test, exp.timewindows[-1])

    # for model in models:
    #     with open(paths['evaluations']['Brier'][model], 'r') as file_:
    #         Brier_results.append(EvaluationResult.from_dict(json.load(file_)))
    Brier = np.array([i.observed_statistic for i in Brier_results])
    brier_label = r'$\mathcal{B}$'
    if isinstance(lowcuts[2], (float, int)):
        Brier[Brier < lowcuts[2]] = lowcuts[2]
    ## Consistency tests
    Consistency_Results = []

    NN = exp._read_results([i for i in exp.tests if i.name == 'Poisson_N'][0], exp.timewindows[-1])
    SS = exp._read_results([i for i in exp.tests if i.name == 'Poisson_S'][0], exp.timewindows[-1])
    MM = exp._read_results([i for i in exp.tests if i.name == 'Poisson_M'][0], exp.timewindows[-1])
    CL = exp._read_results([i for i in exp.tests if i.name == 'Poisson_CL'][0], exp.timewindows[-1])
    for model in models:
    # for n, m, s in zip(paths.get_csep_result('N', years),
    #                    paths.get_csep_result('M', years),
    #                    paths.get_csep_result('S', years)):
        n = [i for i in NN if i.sim_name == model][0]
        m = [i for i in MM if i.sim_name == model][0]
        s = [i for i in SS if i.sim_name == model][0]
        cl = [i for i in CL if i.sim_name == model][0]
        # with open(paths['evaluations']['Poisson_N'][model], 'r') as file_:
        #     n = EvaluationResult.from_dict(json.load(file_))
        # with open(paths['evaluations']['Poisson_S'][model], 'r') as file_:
        #     s = EvaluationResult.from_dict(json.load(file_))
        # with open(paths['evaluations']['Poisson_M'][model], 'r') as file_:
        #     m = EvaluationResult.from_dict(json.load(file_))

        model_cons = []
        if n.quantile[0] > p/2. and n.quantile[1] < 1-p/2.:
            model_cons.append(r'$N$')
        if m.quantile > p:
            model_cons.append('$M$')
        if s.quantile > p:
            model_cons.append('$S$')
        if cl.quantile > p:
            model_cons.append('$CL$')
        Consistency_Results.append(model_cons)

    names = [i.sim_name for i in LL]


    ##  Order results in terms of ~average performance for visualization purpose
    order_val = norm(ll_score) + norm(bs_score) + norm(Brier)
    order = np.flip(np.argsort(order_val))

    ll_score = ll_score[order]
    bs_score = bs_score[order]
    Brier = Brier[order]


    model_names = [names[i] for i in order]
    Consistency_Results = [Consistency_Results[i] for i in order]

    ## Complete array
    colors = ['darkred',
              'darkorange',
              'teal',
              'olivedrab']

    Axes = plot_scores([ll_score, bs_score, Brier],
                       colors[:3],
                       [ll_label, bs_label, brier_label],
                       model_names, format=format, angle_offset=60, offset=10, min_y=0.3, ny=4, fontsize=9,
                       lowbounds=[bool(i) for i in lowcuts])
    Axes = plot_all_consistencies(Axes, Consistency_Results, color=colors[3])

    Axes = plot_legends(Axes, colors,  labels)
    # Axes.set_title('Test results - %i years' % (years) , pad=15, fontsize=14)
    if savepath:
        plt.savefig(savepath, dpi=300,  format='svg', transparent=True)
    plt.show()



if __name__ == '__main__':
    cfg = 'config.yml'

    exp = experiment.Experiment.from_yml(cfg)
    # exp.set_test_date(exp.end_date)
    exp.set_tests()
    exp.set_models()
    # exp.stage_models()
    exp.prepare_paths()
    p = 0.05
    labels = ['Log-Likelihood $\mathcal{L}$',
              'Binomial Score $\mathcal{S}_{B}$',
              'Brier score $\mathcal{B}$',
              'Poisson Consistency $(p\geq%.2f)$' % p]

    plot_results(exp,
                 p=p,
                 labels=labels,
                 format=['%i', '%i', '%.4e',],
                 lowcuts=[-189, -95, -0.00010851],
                 savepath='multiscore.svg')


    # plot_results(5,
    #               labels=labels,
    #               format=['%i', '%i', '%.4e', '%.4e'],
    #               lowcuts=[-140, -140, -0.00005428, -0.00005428],
    #               savepath=paths.get_csep_figpath('Total', 5))