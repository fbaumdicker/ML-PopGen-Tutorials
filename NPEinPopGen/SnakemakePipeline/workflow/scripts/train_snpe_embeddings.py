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
import torch.nn.functional as F
from scipy import stats
from sbi.utils import process_prior,posterior_nn
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, DirectPosterior
from sbi.utils import user_input_checks_utils


def train_and_save_posterior(net, theta, x, save_path):
    neural_posterior = posterior_nn(model="maf", embedding_net=net, hidden_features=10, num_transforms=2)
    inference = SNPE(density_estimator=neural_posterior)
    inference.append_simulations(torch.Tensor(theta), torch.Tensor(x), data_device='cpu')
    inference.train(show_train_summary=True,retrain_from_scratch=True)
    post = inference.build_posterior()

    final_path = os.path.join(snakemake.params.postdir, 'embedding')
    os.makedirs(final_path, exist_ok=True)
    with open(os.path.join(final_path, f'{save_path}_posterior.pkl'), "wb") as pkl:
        pickle.dump(post, pkl)

    return post


if __name__ == '__main__':
    if snakemake.params.use_embedding_net is not True:
        print('not using embedding network')
        sys.exit()

    f = os.path.join(snakemake.params.datadir, "logs", 'embedding_snpe_train_eval.log')
    logging.basicConfig(filename=f, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    simul_param = load_dict_from_json(snakemake.input.simul_param)
    training_param = utils.load_dict_from_json(snakemake.input.train_param)

    std_tr_x = pd.read_csv(os.path.join(snakemake.input.train_val, 'std_train_x.csv'), index_col='scenario')
    std_te_x = pd.read_csv(os.path.join(snakemake.input.test, 'std_test_x.csv'), index_col='scenario')

    logging.info("Loading " + str(len(std_tr_x)) + " training sets, and " + str(len(std_te_x)) + " testing sets.")

    num_output = len(training_param['param_2_learn'])
    device = snakemake.params.device

    net_param = {'num_SNP': training_param['SNP_min'],
                 'num_sample': training_param['sample_min'],
                 'num_output': num_output,
                 'num_block': training_param['num_block'],
                 'num_feature': training_param['num_feature'],
                 'device': device
                 }

    theta = []
    x = []
    excnn_X = []
    for scen in std_tr_x.index:
        for rep in range(snakemake.params.num_rep):
            npz_path = f'{snakemake.params.datadir}/train/checked_SNPs/scenario_{scen}_rep_{rep}.npz'
            with np.load(os.path.join(npz_path)) as data_npz:
                snp = data_npz['SNP']
                pos = data_npz['POS']
                snp, pos = utils.remove_maf_folded(snp, pos, 0)
                input_val = torch.cat((torch.Tensor(pos[:400].astype('float32')).unsqueeze(0),
                                       torch.Tensor(snp[:50, :400].astype('float32'))))
                x.append(input_val)
                theta.append(torch.tensor(std_tr_x.loc[scen].values.astype('float32')))
                excnn_X.append(input_val.unsqueeze(0))

    x = torch.stack(x)
    excnn_X = torch.stack(excnn_X)
    theta = torch.stack(theta)

    if snakemake.params.embedding_net == "EXCNN":
        excnn = ExchangeableCNN(latent_dim=theta.shape[1], channels=excnn_X.shape[1])
        posterior = train_and_save_posterior(excnn, theta, excnn_X, "embedded_EXCNN")
    elif snakemake.params.embedding_net == "SPIDNA":
        spidna = SPIDNA(**net_param).to(device)
        posterior = train_and_save_posterior(spidna, theta, x, "embedded_SPIDNA")

    te_x = []
    for scen in std_te_x.index:
        for rep in range(snakemake.params.num_rep):
            npz_path = f'{snakemake.params.datadir}/test/checked_SNPs/scenario_{scen}_rep_{rep}.npz'
            with np.load(npz_path) as data_npz:
                snp = data_npz['SNP']
                pos = data_npz['POS']
                input_val = torch.cat((torch.Tensor(pos[:400].astype('float32')).unsqueeze(0),
                                       torch.Tensor(snp[:50, :400].astype('float32')))).unsqueeze(0)
                te_x.append(input_val)
    te_x = torch.stack(te_x)

    error = []
    for i, p in enumerate(std_te_x.values):
        tmp_sample = posterior.sample((snakemake.params.sample_num,), x=te_x[i], show_progress_bars=False).median(0)[0]
        err = (tmp_sample - p) ** 2
        error.append(err)
    error = np.stack(error)

    logging.info("The mse of the predictions is " + str(error.mean()))
