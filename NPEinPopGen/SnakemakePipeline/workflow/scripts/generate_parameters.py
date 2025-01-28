"""
This script is designed for generating and simulating demographic scenarios in population genetics.
It handles a range of tasks from generating demographic parameters, simulating populations based on these parameters,
and trimming SNP data. The script is highly configurable through command line arguments, allowing users to specify
paths for input/output data, select specific scenarios for simulation, and control various simulation parameters.

Usage:
    Run this script from the command line with required options. Example:
    $ python generate_parameters.py --data_path "../data" --simulate True

Options:
    --data_path                  Specifies the base directory for output data.
    --scenario_id                ID of the scenario to start simulations from.
    --nb_scenarios_to_simulate   Number of scenarios to simulate.
    --scenario_param_path        Path to a CSV file containing demographic parameters.
    --simul_param_path           Path to a JSON file containing simulation parameters.
    --simulate                   Flag to trigger the simulation process.

Attributes:
    This script contains several functions for data handling and processing, including:
    - Parameter generation and standardization.
    - Demographic simulation using pre-set or user-defined parameters.
    - Data trimming based on SNP count thresholds.
    - Summary statistics computation for demographic analysis.

Author:
    Yuxin Ning (ning.yuxin@student.uni-tuebingen.de)

Version:
    1.0.0

Date:
    May 8, 2024
"""

import argparse
import os
import re
import logging
import numpy as np
# import allel
import msprime
import pandas as pd
import json

from simulatePop import PopulationSizePrior
import simulatePop

from concurrent.futures import ThreadPoolExecutor, as_completed


def load_dict_from_json(filepath):
    """Load a json in a dictionnary.

    The dictionnary contains the overall parameters used for the simulation
    (e.g. path to the data folder, number of epoch). The json can be created
    from a dictionnary using ``save_dict_in_json``.

    Arguments:
        filepath (string): filepath to the json file
    """
    return json.loads(open(filepath).read())


def save_dict_in_json(filepath, param):
    """Save a dictionnary into a json file.

    Arguments:
        filepath (string): filepath of the json file
        param(dict): dictionnary containing the overall parameters used for
        the simulation (e.g. path to the data folder, number of epoch...)
    """
    with open(filepath, 'w') as file_handler:
        json.dump(param, file_handler)


def generate_scenarios_parameters(num_scenario, num_sample, tmax, mutation_rate,
                                  recombination_rate_min, recombination_rate_max,
                                  n_min, n_max, num_replicates, segment_length,
                                  num_time_windows, time_rate, **kwargs):
    """
    Generate demographic parameters for a given number of scenarios.

    Parameters
    ----------
    num_scenario (int): Number of scenarios to generate.
    num_sample (int): Number of samples per scenario.
    tmax (float): Maximum time for population events.
    mutation_rate (float): Mutation rate to be used across all scenarios.
    recombination_rate_min (float): Minimum recombination rate.
    recombination_rate_max (float): Maximum recombination rate.
    n_min (int): Minimum population size.
    n_max (int): Maximum population size.
    num_replicates (int): Number of replicates per scenario.
    segment_length (float): Length of the genomic segment to be simulated.
    num_time_windows (int): Number of time windows for population size changes.
    time_rate (float): Rate at which time changes across windows.

    Returns
    -------
    pandas.DataFrame: DataFrame containing all the parameters for each scenario.
    """
    scenario_param = pd.DataFrame(columns=['scenario', 'mutation_rate',
                                           'recombination_rate', 'num_replicates',
                                           'num_sample', 'segment_length'])
    scenario_param['scenario'] = np.array(range(num_scenario))
    scenario_param['mutation_rate'] = np.full(num_scenario, mutation_rate)
    scenario_param['num_replicates'] = np.full(num_scenario, num_replicates)
    scenario_param['num_sample'] = np.full(num_scenario, num_sample)
    scenario_param['segment_length'] = np.full(num_scenario, segment_length)

    prior = PopulationSizePrior(num_time_windows, n_min, n_max, recombination_rate_min, recombination_rate_max,
                                return_numpy=True)
    samples = prior.sample((num_scenario,))

    # Extract population sizes and recombination rates from samples.
    population_size = samples[..., :-1]
    scenario_param['recombination_rate'] = samples[..., -1]

    population_size = pd.DataFrame(np.array(population_size).astype(int),
                                   columns=[f'population_size_{i}' for i in range(num_time_windows)])

    population_time = np.repeat([[(np.exp(np.log(1 + time_rate * tmax) * i /
                                          (num_time_windows - 1)) - 1) / time_rate for i in
                                  range(num_time_windows)]], num_scenario, axis=0)
    population_time = pd.DataFrame(np.around(population_time).astype(int),
                                   columns=[f'population_time_{i}' for i in range(num_time_windows)])

    scenario_param = pd.concat([scenario_param, population_time, population_size], axis=1, sort=False)
    scenario_param = scenario_param.set_index('scenario')
    return scenario_param


def extract_numbers(file_name, pattern=r"scenario_(\d+).npz"):
    """
    Extract scenario number from the filename using a regex pattern.

    Parameters
    ----------
    file_name (str): Filename from which to extract the number.
    pattern (str, optional): Regex pattern to match the scenario number. Default is "scenario_(\d+).npz".

    Returns
    -------
    The extracted number if a match is found, otherwise None.
    """
    match = re.search(pattern, file_name)
    if match:
        return int(match.group(1))
    else:
        return None


def generate_data(parameters, simulation_data_path, num_time_windows, num_sample, num_replicates, segment_length,
                  max_cores, **kwargs):
    """
    Generate SNP data by simulating population based on specified parameters.

    Parameters
    ----------
    parameters (pandas.DataFrame): Parameters for simulation including population sizes and times, and recombination
        rate.
    data_path (str): Path to store the generated SNP data.
    num_time_windows (int): Number of time windows for population dynamics.
    num_sample (int): Number of samples to simulate.
    num_replicates (int): Number of replicates to simulate.
    segment_length (float): Length of the genomic segment to simulate.
    """
    print('The simulation will be stored into ', simulation_data_path)
    # extract only the needed pop sizes and recombination rate
    time_col = [f'population_time_{i}' for i in range(num_time_windows)]
    pop_time = parameters[time_col].values[0]

    parameters = parameters.reset_index()
    cols = ['scenario'] + [f'population_size_{i}' for i in range(num_time_windows)] + ['recombination_rate']
    true_pop = parameters[cols].values

    print('generating SNP data')
    simulatePop.simulate_population(true_pop, simulation_data_path, num_sample=num_sample, SNP_min=400,
                                    num_rep=num_replicates,
                                    population_time=pop_time, segment_length=segment_length, max_cores=max_cores)
    print('SNPs data stored')


def trim_data(data_path, parameter, path_to_SNP, num_time_windows):
    """
    Trim data based on available SNP files and store the processed population sizes.

    Parameters
    ----------
    data_path (str): Path where the trimmed data will be stored.
    parameter (pandas.DataFrame): DataFrame containing scenario parameters.
    path_to_SNP (str): Directory containing SNP data files.
    num_time_windows (int): Number of time windows for population dynamics.
    """
    file_names = os.listdir(path_to_SNP)
    index_list = np.sort([extract_numbers(file_name) for file_name in file_names])
    selected_rows = parameter.loc[index_list]
    cols = [f'population_size_{i}' for i in range(num_time_windows)] + ['recombination_rate']
    trimmed_pop = selected_rows[cols]
    trimmed_pop.to_csv(os.path.join(data_path, 'preprocessed_data/filtered_population_sizes.csv'), index='scenario')

    print('trimmed population sizes stored')


if __name__ == "__main__":
    os.makedirs(snakemake.params.data_path, exist_ok=True)
    simul_param = {'num_scenario': snakemake.params.num_sim, 'num_sample': 50, 'tmax': 130000,
                   'recombination_rate_min': 1e-9,
                   'recombination_rate_max': 1e-8, 'mutation_rate': 1e-8,
                   'n_min': 10, 'n_max': 100000, 'num_replicates': snakemake.params.num_rep,
                   'segment_length': 2e6, 'model_name': snakemake.params.model,
                   'data_path': snakemake.params.data_path, 'seed_simulation': 2,
                   'num_time_windows': 21, 'time_rate': 0.06,
                   'simulation_data_path':snakemake.params.data_path}

    json_filename = simul_param['model_name'] + '_simul_parameters.json'

    # Ensure the directory exists and save the simulation parameters as a JSON file.
    os.makedirs(simul_param['simulation_data_path'], exist_ok=True)
    save_dict_in_json(os.path.join(simul_param['simulation_data_path'], json_filename), simul_param)

    # Generate a csv file with all demographic parameters
    scenario_param = generate_scenarios_parameters(**simul_param)
    os.makedirs(simul_param['simulation_data_path'], exist_ok=True)
    scenario_param.to_csv(os.path.join(simul_param['simulation_data_path'], simul_param['model_name'] +
                                       '_demo_parameters.csv'), index_label='scenario')

    generate_data(parameters=scenario_param, max_cores=snakemake.params.max_cores, **simul_param)
    os.makedirs(os.path.join(simul_param['simulation_data_path'], 'preprocessed_data'), exist_ok=True)
    trim_data(simul_param['simulation_data_path'],
              scenario_param, os.path.join(simul_param['simulation_data_path'], 'SNP400'),
              simul_param['num_time_windows'])
