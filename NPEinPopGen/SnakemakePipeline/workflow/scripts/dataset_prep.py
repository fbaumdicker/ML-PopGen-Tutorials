import pandas as pd
import os
import utils
import numpy as np


def data_prep(training_param, sumstat_path):
    """ Prepare and preprocess data by loading, cleaning, and splitting into training and testing sets. """

    # load training parameters
    scenario_param = pd.read_csv(training_param['preprocessed_scenario_param_path'], index_col='scenario')
    del scenario_param['recombination_rate']

    sumstat = pd.read_csv(sumstat_path, index_col='scenario')
    common_indices = scenario_param.index.intersection(sumstat.index)
    sumstat = sumstat.loc[common_indices].sort_index()
    scenario_param = scenario_param.loc[common_indices].sort_index()

    train_pop = scenario_param.loc[scenario_param['training_set']]
    val_pop = scenario_param.loc[scenario_param['training_set'] == False]
    train_sum = sumstat.loc[train_pop.index]
    val_sum = sumstat.loc[val_pop.index]
    del train_pop['training_set'], val_pop['training_set']

    return train_pop, val_pop, train_sum, val_sum


if __name__ == '__main__':
    training_path = os.path.join(snakemake.params.datadir, snakemake.params.model, 'dataset_4_inference')
    os.makedirs(training_path, exist_ok=True)
    train_param = utils.load_dict_from_json(snakemake.input.train_param)

    if snakemake.params.model == "train":
        standard_train_x, standard_val_x, train_y, val_y = data_prep(train_param, snakemake.input.sum_stat)
        standard_train_x.to_csv(os.path.join(training_path, 'std_train_x.csv'))
        train_y.to_csv(os.path.join(training_path, 'train_y.csv'))
        standard_val_x.to_csv(os.path.join(training_path, 'std_val_x.csv'))
        val_y.to_csv(os.path.join(training_path, 'val_y.csv'))
    else:
        sumstat = pd.read_csv(snakemake.input.sum_stat, index_col='scenario')
        pop = pd.read_csv(snakemake.input.population, index_col='scenario').apply(np.log)
        del pop['recombination_rate']
        tmean = train_param["train_mean"]
        tstd=train_param["train_std"]
        pop = (pop - tmean) / tstd
        sumstat.to_csv(os.path.join(training_path, 'test_y.csv'))
        pop.to_csv(os.path.join(training_path, 'std_test_x.csv'))
