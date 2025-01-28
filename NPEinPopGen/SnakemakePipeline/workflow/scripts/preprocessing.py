#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import sys
import numpy as np
import pandas as pd
import utils


class TrainingParam:
    def __init__(self, training_param_path=None, scenario_param_path=snakemake.input.population_sizes,
                 simulation_param_path=snakemake.input.simul_param, num_validation_scenarios=snakemake.params.num_vali,
                 num_training_scenarios=snakemake.params.num_train, run_path=snakemake.params.datadir,
                 param_2_learn=None, param_2_log=None, SNP_min=400, use_cuda=True, cuda_device=None,
                 num_epoch=2, learning_rate=0.001, weight_decay=0, batch_size=100,
                 evaluation_interval=100000, network_name=None,
                 seed=2, loader_num_workers=20, num_block=7, num_feature=50,
                 sample_min=50, train_mean=None, train_std=None, maf=0, transform_allel_min_major=False,
                 start_from_last_checkpoint=False, alpha=None, **kwargs):
        if param_2_learn is None:
            param_2_learn = ['population_size_' + str(i) for i in range(21)]
        if training_param_path is None:
            for key, value in locals().items():
                if key != 'self' and key != 'training_param_path' and key != 'kwargs':
                    setattr(self, key, value)
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            self.load(training_param_path)

    def save(self, training_param_path):
        logging.info(f'Saving training parameters at: {training_param_path}')
        utils.save_dict_in_json(training_param_path, vars(self))

    def load(self, training_param_path):
        training_param_dict = utils.load_dict_from_json(training_param_path)
        for key, value in training_param_dict.items():
            setattr(self, key, value)

    def preprocess(self):
        simulation_param = utils.load_dict_from_json(self.simulation_param_path)
        np.random.seed(self.seed)
        try:
            scenario_param = pd.read_csv(self.scenario_param_path,
                                         index_col='scenario',  # should have index as first column
                                         sep=None,  # autodetect separator
                                         engine='python')  # to use sep=None
        except ValueError as e:
            if 'scenario' in str(e):
                logging.error(f'A columns scenario_idx must exist in your table {self.scenario_param_path}.')
            else:
                logging.error(e)
            sys.exit(0)
        if self.param_2_learn is None:
            self.param_2_learn = [param for param in list(scenario_param) if 'size' in param or 'time' in param]
            logging.info(
                f'No parameters to learn provided, using parameters with "size" or "time" in their name: {self.param_2_learn}')
        scenario_param = scenario_param.sort_index().iloc[0:self.num_training_scenarios + self.num_validation_scenarios]
        logging.info(f'Removing scenario without full or with less than {self.SNP_min} SNPs...')

        logging.info('Splitting scenarios between training and validation set.')
        val = np.tile(False, self.num_validation_scenarios)
        if scenario_param.shape[0] - self.num_validation_scenarios > 0:
            train = np.tile(True, scenario_param.shape[0] - self.num_validation_scenarios)
        else:
            logging.error('num_validation_scenarios is greater than the number of scenario.')
            raise ValueError
        train_set = np.concatenate((val, train))
        np.random.shuffle(train_set)
        scenario_param['training_set'] = train_set
        if self.param_2_log is None:
            self.param_2_log = [param for param in list(scenario_param) if 'size' in param]
            logging.info(
                f'No parameters to log transform have been provided, log-transforming parameters with "size" in their name:{self.param_2_log}')
        if self.param_2_log != []:
            logging.info('Log-transform demographic parameters.')
            scenario_param[self.param_2_log] = scenario_param[self.param_2_log].apply(np.log)

        logging.info('Standardize demographic parameters.')
        train_param = scenario_param.loc[scenario_param['training_set']]
        train_param = train_param.drop('training_set', axis=1)
        train_mean = train_param[self.param_2_learn].mean()
        train_std = train_param[self.param_2_learn].std()
        scenario_param[self.param_2_learn] = (
                (scenario_param[self.param_2_learn] - train_mean) / train_std)

        self.train_mean = train_mean.to_dict()
        self.train_std = train_std.to_dict()
        # Save training parameters and preprocessed scenario parameters
        scenario_param_filename = f'{simulation_param["model_name"]}_preprocessed_param.csv'
        logging.info(
            f'Saving preprocessed scenario parameters at: {os.path.join(self.run_path, scenario_param_filename)}')
        scenario_param_path = os.path.join(self.run_path, scenario_param_filename)
        scenario_param.to_csv(scenario_param_path, index_label='scenario')
        self.preprocessed_scenario_param_path = scenario_param_path
        training_param_path = os.path.join(self.run_path, f'{simulation_param["model_name"]}_training_param.json')

        self.save(training_param_path)
        return training_param_path


if __name__ == '__main__':
    training_param = TrainingParam()
    training_param.save(os.path.join(snakemake.params.datadir, 'training_param.json'))
    training_param.preprocess()


