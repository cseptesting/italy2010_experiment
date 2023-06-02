import os
from floatcsep.experiment import Experiment
from floatcsep.utils import timewindow2str
from k_function import k_ripley_test, ripley2hdf5

def main():

    cfg_file = os.path.join('config.yml')
    experiment = Experiment.from_yml(cfg_file)
    experiment.stage_models()
    time_window = timewindow2str(experiment.timewindows[0])
    models = [i.get_forecast(time_window) for i in experiment.models]
    os.makedirs('results', exist_ok=True)

    for model in models:
        res = k_ripley_test(model, experiment.catalog,
                            nsim=100,
                            r_disc=100)
        ripley2hdf5(res, os.path.join('results', f'K_{model.name}.hdf5'))


if __name__ == '__main__':
    main()
