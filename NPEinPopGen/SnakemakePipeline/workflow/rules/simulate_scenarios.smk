import os.path
import pandas as pd

from common import count_scenario

configfile: "config/neutral_simulation.yaml"

rule simulate_filtration:
    message: "Simulating data for {wildcards.model}"
    output:
        param_file=os.path.join(config['datadir'], "{model}", "{model}_simul_parameters.json"),
        demo_file=os.path.join(config['datadir'], "{model}", "{model}_demo_parameters.csv"),
        checked_snps=directory(os.path.join(config['datadir'], "{model}", 'checked_SNPs')),
        snp400=directory(os.path.join(config['datadir'], "{model}", 'SNP400')),
        population=os.path.join(config['datadir'], "{model}", "preprocessed_data/filtered_population_sizes.csv")
    conda:
        "sbi38"
    log:
        os.path.join(config['datadir'], "logs", "{model}_simulate.log")
    params:
        num_sim=lambda wildcards: config['model'][wildcards.model]['num_sim'],
        simulate=True,
        max_cores=3,
        num_rep=config['num_rep'],
        data_path=lambda wildcards, output: os.path.dirname(output.param_file),
        model=lambda wildcards: wildcards.model
    script:
        "../scripts/generate_parameters.py"


rule summary_stats:
    message: "Compute summary statistics for {wildcards.model}"
    input:
        checked_snps=os.path.join(config['datadir'],'{model}','checked_SNPs')
    output:
        sumstat=os.path.join(config['datadir'],'{model}',"preprocessed_data/sum_stat.csv")
    params:
        # datadir = lambda wildcards, output: os.path.dirname(output.sumstat),
        datadir=os.path.join(config['datadir'],'{model}'),
        max_cores=3,
        replication=config['num_rep']
    conda:
        "sbi38"
    log:
        os.path.join(config['datadir'],"logs","{model}_sumstat.log")
    script:
        "../scripts/summary_statistics.py"


rule standardization_params:
    message: "Data preparation for following methods for {wildcards.model}"
    input:
        population_sizes=os.path.join(config['datadir'],"{model}","preprocessed_data","filtered_population_sizes.csv"),
        simul_param=os.path.join(data_path,"{model}","{model}_simul_parameters.json"),
    output:
        train_param=os.path.join(config['datadir'],"{model}","training_param.json"),
        param=os.path.join(data_path,"{model}","{model}_training_param.json"),
        preprocessed_data=os.path.join(data_path,"{model}","{model}_preprocessed_param.csv")
    conda:
        "sbi38"
    log:
        os.path.join(config['datadir'],"logs","{model}_preprocess.log")
    params:
        datadir=lambda wildcards, output: os.path.dirname(output.train_param),
        # datadir=os.path.join(config['datadir'],'{model}'),
        num_vali=1,
        num_train=lambda wildcards, input: count_scenario(input.population_sizes)
    wildcard_constraints:
        model="train"
    script:
        "../scripts/preprocessing.py"

rule prepare_dataset:
    message: "Store data sets from {wildcards.model} for further inference methods"
    input:
        train_param=os.path.join(config['datadir'],"train", "train_training_param.json"),
        sum_stat=os.path.join(config['datadir'],'{model}',"preprocessed_data/sum_stat.csv"),
        population=os.path.join(config['datadir'],"{model}","preprocessed_data/filtered_population_sizes.csv")
    output:
        dataset = directory(os.path.join(config['datadir'], "{model}","dataset_4_inference"))
    conda:
        "sbi38"
    log:
        os.path.join(config['datadir'],"logs","{model}_directsave.log")
    params:
        # datadir = lambda wildcards, output: os.path.dirname(output.dataset),
        datadir=config["datadir"],
        model="{model}"
    script:
        "../scripts/dataset_prep.py"