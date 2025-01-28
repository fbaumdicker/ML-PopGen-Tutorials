import os.path
import argparse
import pandas as pd
import numpy as np
# from sbi import utils as utils

import utils
from sbi import analysis as analysis
import torch
import pickle
from simulatePop import PopulationSizePrior
from generate_parameters import load_dict_from_json
import sys
from net import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import logging

from scipy import stats
from sbi.utils import process_prior
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, DirectPosterior
from sbi.utils import user_input_checks_utils


def train_and_save_posterior(theta, x, save_path):
    infer = SNPE()
    infer.append_simulations(theta, x,data_device='cpu').train(show_train_summary=True, retrain_from_scratch=True)
    post = infer.build_posterior()

    final_path = os.path.join(snakemake.params.postdir, save_path)
    os.makedirs(final_path, exist_ok=True)
    with open(os.path.join(final_path, 'posterior.pkl'), "wb") as pkl:
        pickle.dump(post, pkl)

    return post


if __name__ == '__main__':
    f = os.path.join(snakemake.params.datadir, "logs", 'sumstat_sbi_train_eval.log')
    logging.basicConfig(filename=f, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    simul_param = load_dict_from_json(snakemake.input.simul_param)
    training_param = utils.load_dict_from_json(snakemake.input.train_param)

    std_tr_x = pd.read_csv(os.path.join(snakemake.input.train_val, 'std_train_x.csv'), index_col='scenario')
    tr_y = pd.read_csv(os.path.join(snakemake.input.train_val, 'train_y.csv'), index_col='scenario')
    # std_val_x = pd.read_csv(os.path.join(snakemake.input.train_val, 'std_val_x.csv'), index_col='scenario')
    # val_y = pd.read_csv(os.path.join(snakemake.input.train_val, 'val_y.csv'), index_col='scenario')
    std_te_x = pd.read_csv(os.path.join(snakemake.input.test, 'std_test_x.csv'), index_col='scenario')
    te_y = pd.read_csv(os.path.join(snakemake.input.test, 'test_y.csv'), index_col='scenario')

    # logging.info("Loading " + str(len(std_tr_x)) + " training sets, " + str(
    #     len(std_val_x)) + " validation sets, and " + str(len(std_te_x)) + " testing sets.")

    logging.info("Loading " + str(len(std_tr_x)) + " training sets, and " + str(len(std_te_x)) + " testing sets.")

    device = snakemake.params.device

    posterior = train_and_save_posterior(torch.Tensor(std_tr_x.values), torch.Tensor(tr_y.values), "sumstat")

    error = []
    for i, p in enumerate(std_te_x.values):
        tmp_sample = posterior.sample((snakemake.params.sample_num,), x=te_y.values[i], show_progress_bars=False).median(0)[0]
        err = (tmp_sample - p) ** 2
        error.append(err)
    error = np.stack(error)

    logging.info("The mse of the predictions is " + str(error.mean()))
