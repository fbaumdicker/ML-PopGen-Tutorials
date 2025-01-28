import msprime

import argparse
import torch
import os
import pandas as pd
import numpy as np
from scipy import stats
from sbi import utils as utils
import matplotlib.pyplot as plt
import allel
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


class PopulationSizePrior:
    """
    Class for modeling population size and recombination rate priors across different time windows.

    Attributes:
        num_time_windows (int): Number of time windows to model.
        min_pop_size (float): Minimum population size.
        max_pop_size (float): Maximum population size.
        min_recomb_rate (float): Minimum recombination rate.
        max_recomb_rate (float): Maximum recombination rate.
        dist (torch.distributions.Distribution): Distribution for population size across time windows.
        return_numpy (bool): Whether to return results as numpy arrays instead of torch tensors.

    Methods:
        __init__: Initializes the class with the defined parameters.
        log_interval: Calculates the log of the population size interval around a given size.
        sample: Samples from the prior distributions for population size and recombination rates.
        log_prob: Calculates the log probability for a given set of population sizes and recombination rates.
    """

    def __init__(self, num_time_windows, min_pop_size, max_pop_size, min_recomb_rate, max_recomb_rate,
                 return_numpy: bool = False):
        self.num_time_windows = num_time_windows
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.min_recomb_rate = min_recomb_rate
        self.max_recomb_rate = max_recomb_rate
        self.dist = utils.BoxUniform(np.log10(min_pop_size) * torch.ones(num_time_windows + 1),
                                     np.log10(max_pop_size) * torch.ones(num_time_windows + 1))
        self.return_numpy = return_numpy

    # Method to calculate the logarithm of the interval of possible population sizes around a given value N.
    def log_interval(self, N):
        # TODO: need to be fixed because the sampling method is different
        I_1 = max(self.min_pop_size, N * 10 ** -1)
        I_2 = min(self.max_pop_size, N * 10 ** 1)
        return torch.log(torch.tensor(I_2 - I_1))

    def sample(self, sample_shape=torch.Size([])):
        recomb_rate_prior = utils.BoxUniform(low=self.min_recomb_rate * torch.ones(1),
                                             high=self.max_recomb_rate * torch.ones(1))
        samples = 10 ** self.dist.sample(sample_shape)
        num_samp = len(sample_shape)

        n_min_log10 = np.log10(self.min_pop_size)

        if num_samp == 0:
            recomb_rate = recomb_rate_prior.sample()
            # generate population sizes
            for j in range(1, self.num_time_windows):
                new_value = 10 ** n_min_log10 - 1
                while new_value > self.max_pop_size or new_value < self.min_pop_size:
                    new_value = samples[j - 1] * 10 ** np.random.uniform(-1, 1)
                samples[j] = new_value
            samples[-1] = recomb_rate
        else:
            for i in range(sample_shape[0]):
                recomb_rate = recomb_rate_prior.sample()
                for j in range(1, self.num_time_windows):
                    new_value = 10 ** n_min_log10 - 1
                    while new_value > self.max_pop_size or new_value < self.min_pop_size:
                        new_value = samples[i, j - 1] * 10 ** np.random.uniform(-1, 1)
                    samples[i, j] = new_value
                samples[i, -1] = recomb_rate
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        # TODO: This prob function is not correct. To be updated.
        if self.return_numpy:
            values = torch.as_tensor(values)
        dim = len(values.shape)
        pop = values[..., :-1]
        recomb = values[..., -1]

        if (pop >= self.min_pop_size).all() and (pop <= self.max_pop_size).all():
            if (recomb >= self.min_recomb_rate).all() and (recomb <= self.max_recomb_rate).all():
                if dim == 1:
                    log_prob = torch.log(torch.tensor(self.max_pop_size - self.min_pop_size))
                    for ps in pop[1:]:
                        log_prob += self.log_interval(ps)
                    log_prob += torch.log(torch.tensor(self.max_recomb_rate - self.min_recomb_rate))
                else:
                    log_prob = torch.empty(torch.Size([values.shape[0]]))
                    for i in range(values.shape[0]):
                        log_prob[i] = torch.log(torch.tensor(self.max_pop_size - self.min_pop_size))
                        for ps in pop[i][1:]:
                            log_prob[i] += self.log_interval(ps)
                        log_prob[i] += torch.log(torch.tensor(self.max_recomb_rate - self.min_recomb_rate))
            else:
                log_prob = torch.tensor(float('-inf'))
        else:
            log_prob = torch.tensor(float('-inf'))

        return log_prob.numpy() if self.return_numpy else log_prob
        # return


def SNP_afs(ts, SNP_min, data_path, scenario):
    """
    Analyze Single Nucleotide Polymorphisms (SNPs) from a set of tree sequences and filter them based on a minimum
    count threshold.

    Parameters
    ----------
    ts (iterator of msprime.Tree): An iterator over tree sequences generated by msprime or similar tool.
    SNP_min (int): Minimum number of SNPs required to consider the data valid.
    data_path (str): Path to the directory where SNP data should be stored.
    scenario (int): Identifier for the scenario under which the simulation is run, used for file naming.

    Returns
    -------
    tuple of lists: Returns a tuple containing two lists, one with SNP arrays and one with positions, if the minimum
    SNP count is met; otherwise, returns None.
    """

    SNPs_of_one_pop = []
    POSs_of_one_pop = []
    for rep, tree in enumerate(ts):
        mts = msprime.sim_mutations(tree, rate=1e-8)
        snp, pos = get_SNPs(mts)
        if len(pos) < SNP_min:
            return None
        else:
            SNPs_of_one_pop.append(snp)
            POSs_of_one_pop.append(pos)

    for idx, (snp, pos) in enumerate(zip(SNPs_of_one_pop, POSs_of_one_pop)):
        np.savez_compressed(os.path.join(data_path, f'checked_SNPs/scenario_{scenario}_rep_{idx}'), SNP=snp, POS=pos)
    return SNPs_of_one_pop, POSs_of_one_pop


def simulate_scenario(p, data_path, num_sample, SNP_min, num_rep, population_time, segment_length):
    """
    Simulate population for a single set of parameters and generate SNP data, saving results only for scenarios that
    meet SNP minimums.

    Parameters
    ----------
    p (list): A list containing population size and recombination rate data for a particular scenario.
    data_path (str): Base directory to store output data.
    num_sample (int): Number of samples to simulate.
    SNP_min (int): Minimum number of SNPs required to save data.
    num_rep (int): Number of replicates to simulate.
    population_time (list): Times at which population parameters change.
    segment_length (float): Length of the genomic segment to simulate.
    """
    demography = msprime.Demography()

    idx = int(p[0])
    pop_sizes = p[1:-1]
    recomb_rate = p[-1]
    demography.add_population(initial_size=pop_sizes[0])

    for i in range(1, len(pop_sizes)):
        demography.add_population_parameters_change(time=population_time[i], initial_size=pop_sizes[i], growth_rate=0)

    ts = msprime.sim_ancestry(
        num_sample,
        sequence_length=segment_length,
        ploidy=1,
        num_replicates=num_rep,
        demography=demography,
        recombination_rate=recomb_rate)

    # Check SNP data and save to files if it meets the SNP_min requirement.
    passed_SNP = SNP_afs(ts, SNP_min, data_path, idx)
    if passed_SNP is not None:
        SNP_400 = np.stack([s[:, :SNP_min] for s in passed_SNP[0]])
        POS_400 = np.stack([s[:SNP_min] for s in passed_SNP[1]])
        np.savez_compressed(os.path.join(data_path, f'SNP400/scenario_{idx}'), SNP=SNP_400, POS=POS_400)


def simulate_population(param, data_path, num_sample, SNP_min, num_rep, population_time, segment_length, max_cores):
    """
    Simulate population based on a set of parameters and generate SNP data, saving results only for scenarios that
    meet SNP minimums.

    Parameters
    ----------
    param (list): List of parameters where each element is a list containing population size and recombination rate
        data for a particular scenario.
    data_path (str): Base directory to store output data.
    num_sample (int): Number of samples to simulate.
    SNP_min (int): Minimum number of SNPs required to save data.
    num_rep (int): Number of replicates to simulate.
    population_time (list): Times at which population parameters change.
    segment_length (float): Length of the genomic segment to simulate.
    max_cores (int): Maximum number of threads to use.
    """
    os.makedirs(os.path.join(data_path, 'checked_SNPs'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'SNP400'), exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_cores) as executor:
        futures = [
            executor.submit(simulate_scenario, p, data_path, num_sample, SNP_min, num_rep, population_time,
                            segment_length)
            for p in param
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")


def get_SNPs(mts):
    """
    Extract SNP information and their positions from a given msprime/mutated tree sequence.

    Parameters
    ----------
    mts(msprime.MutationTreeSequence): The mutated tree sequence object from which to extract SNPs.

    Returns
    -------

    """
    positions = [variant.site.position for variant in mts.variants()]
    positions = np.array(positions) - np.array([0] + positions[:-1])
    positions = positions.astype(int)

    SNPs = mts.genotype_matrix().T.astype(np.uint8)
    return SNPs, positions
