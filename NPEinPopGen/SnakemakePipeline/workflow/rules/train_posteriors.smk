import os.path

configfile: "config/neutral_simulation.yaml"

rule SNPE_on_summary_stats:
    message: "train SNPE using summary statistics"
    input:
        train_param=os.path.join(config['datadir'],"train/train_training_param.json"),
        train_val=os.path.join(config['datadir'], "train/dataset_4_inference"),
        test=os.path.join(config['datadir'], "test/dataset_4_inference"),
        simul_param=os.path.join(config['datadir'],"train/train_simul_parameters.json")
    output:
        posteriors=directory(os.path.join(config["posteriordir"], "sumstat"))
    log:
        os.path.join(config["datadir"],"logs",'sumstat_snpe_train_eval.log')
    conda:
        "sbi38"
    params:
        datadir=config["datadir"],
        postdir=config["posteriordir"],
        max_cores=3,
        device= 'cpu',
        num_rep = config['num_rep'],
        sample_num= config['sample'],
        sum_stat = config['use_summary_statistics']
    script:
        "../scripts/train_snpe_sumstat.py"

rule SNPE_with_embeddingNN:
    message: "train SNPE using embedding networks"
    input:
        train_param=os.path.join(config['datadir'],"train/train_training_param.json"),
        train_val=os.path.join(config['datadir'],"train/dataset_4_inference"),
        test=os.path.join(config['datadir'],"test/dataset_4_inference"),
        simul_param=os.path.join(config['datadir'],"train/train_simul_parameters.json")
    output:
        posteriors=directory(os.path.join(config["posteriordir"], "embedding"))
    log:
        os.path.join(config["datadir"],"logs",'embedding_snpe_train_eval.log')
    conda:
        "sbi38"
    params:
        datadir=config["datadir"],
        postdir=config["posteriordir"],
        max_cores=3,
        device = 'cpu',
        sample_num=config['sample'],
        num_rep= config['num_rep'],
        use_embedding_net=config['use_embedding_net'],
        embedding_net=config['embedding_net']
    script:
        "../scripts/train_snpe_embeddings.py"