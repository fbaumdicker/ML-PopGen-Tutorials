import os
import re
import logging
import numpy as np
import allel
import pandas as pd
from simulatePop import PopulationSizePrior
import simulatePop

import argparse
from multiprocessing import Pool, cpu_count


def extract_numbers(file_name, pattern=r"scenario_(\d+)_rep_(\d+).npz"):
    '''
    get scenario and rep from each SNP file
    :param file_name: name of certain SNP file
    :param pattern: the format of file name
    :return: scenario, rep
    '''
    match = re.search(pattern, file_name)
    if match:
        scenario = int(match.group(1))
        replication = int(match.group(2))
        return scenario, replication
    else:
        return None


def LD(haplotype, pos_vec, size_chr, circular=True, distance_bins=None):
    """
    Compute LD for a subset of SNPs drawn with different gap sizes in between them.
    Gap sizes follow power 2 distribution.
    The LD is then computed and averaged over different bin (distance_bins) sizes.

    Parameters ---------- haplotype : numpy 2D array or allel.haplotype SNP matrix where in the first dimension are
    the SNP (rows) and in the second dimension (columns) are the samples. pos_vec : 1D array array of absolute
    positions in [0, size_chr]. size_chr : int Size of the chromosome. circular : bool Whether to consider the
    chromosome circular or not. If circular, the maximum distance between 2 SNPs is thus half the chromosome.
    distance_bins : int or list LD will be averaged by bins of distances e.g. if distance_bins = [0, 100, 1000,
    10000], LD will be averaged for the groups [0,100[, [100, 1000[, and [1000, 10000[ If distance_bins is an int,
    it defines the number of bins of distances for which to compute the LD The bins are created in a logspace If
    distance_bins is a list, they will be used instead

    Returns
    -------
    DataFrame
        Table with the distance_bins as index, and the mean value of
    """
    if distance_bins is None or isinstance(distance_bins, int):
        if isinstance(distance_bins, int):
            n_bins = distance_bins - 1
        else:
            n_bins = 19
        if circular:
            distance_bins = np.logspace(2, np.log10(size_chr // 2), n_bins)
            distance_bins = np.insert(distance_bins, 0, [0])
        else:
            distance_bins = np.logspace(2, np.log10(size_chr), n_bins)
            distance_bins = np.insert(distance_bins, 0, [0])

    # Iterate through gap sizes
    n_SNP, n_samples = haplotype.shape
    gaps = (2 ** np.arange(0, np.log2(n_SNP), 1)).astype(int)

    # Initialize lists to store selected SNP pairs and LD values
    selected_snps = []
    for gap in gaps:
        snps = np.arange(0, n_SNP, gap) + np.random.randint(0, (n_SNP - 1) % gap + 1)
        # adding a random start (+1, bc 2nd bound in randint is exlusive)

        # non overlapping contiguous pairs
        # snps=[ 196, 1220, 2244] becomes
        # snp_pairs=[(196, 1220), (1221, 2245)]
        snp_pairs = np.unique([((snps[i] + i) % n_SNP, (snps[i + 1] + i) % n_SNP) for i in range(len(snps) - 1)],
                              axis=0)

        # If we don't have enough pairs (typically when gap is large), we add a random rotation until we have at
        # least 300) count = 0

        if not circular:
            snp_pairs = snp_pairs[snp_pairs[:, 0] < snp_pairs[:, 1]]
        last_pair = snp_pairs[-1]

        if circular:
            max_value = n_SNP - 1
        else:
            max_value = n_SNP - gap - 1

        while len(snp_pairs) <= min(300, max_value):
            # count += 1 if count % 10 == 0: print(">>  " + str(gap) + " - " + str(len(np.unique(snp_pairs,
            # axis=0))) + " -- "+ str(len(snps) - 1) + "#" + str(count)) remainder = (n_SNP - 1) % gap if (n_SNP - 1)
            # % gap != 0 else (n_SNP - 1) // gap
            random_shift = np.random.randint(1, n_SNP) % n_SNP
            new_pair = (last_pair + random_shift) % n_SNP
            snp_pairs = np.unique(np.concatenate([snp_pairs, new_pair.reshape(1, 2)]), axis=0)
            last_pair = new_pair

            if not circular:
                snp_pairs = snp_pairs[snp_pairs[:, 0] < snp_pairs[:, 1]]

        selected_snps.append(snp_pairs)

    # Functions to aggregate the values within each distance bin
    agg_bins = {"snp_dist": ["mean"], "r2": ["mean", "count", "sem"]}

    ld = pd.DataFrame()
    for i, snps_pos in enumerate(selected_snps):

        if circular:
            sd = pd.DataFrame((np.diff(pos_vec[snps_pos]) % size_chr) % (size_chr // 2),
                              columns=["snp_dist"])  # %size_chr/2 because max distance btw 2 SNP is size_chr/2
        else:
            sd = pd.DataFrame((np.diff(pos_vec[snps_pos])), columns=["snp_dist"])

        sd["dist_group"] = pd.cut(sd.snp_dist, bins=distance_bins)
        sr = [allel.rogers_huff_r(snps) ** 2 for snps in haplotype[snps_pos]]
        sd["r2"] = sr
        sd["gap_id"] = i
        ld = pd.concat([ld, sd])

    ld2 = ld.dropna().groupby("dist_group", observed=False).agg(agg_bins)

    # Flatten the MultiIndex columns and rename explicitly
    ld2.columns = ['_'.join(col).strip() for col in ld2.columns.values]
    ld2 = ld2.rename(columns={
        'snp_dist_mean': 'mean_dist',
        'r2_mean': 'mean_r2',
        'r2_count': 'Count',
        'r2_sem': 'sem_r2'
    })
    # ld2 = ld2.fillna(-1)
    return ld2[['mean_dist', 'mean_r2', 'Count', 'sem_r2']]


def sfs(haplotype, ac):
    """
    Calculate the site frequency spectrum (SFS) from haplotype data and allele counts.

    Parameters
    ----------
    haplotype (numpy.ndarray): The haplotype matrix where rows represent variants and columns represent individuals.
    ac (numpy.ndarray): Allele count array where each entry represents the count of the derived allele at a site.

    Returns
    -------
    pandas.DataFrame: DataFrame containing the SFS. Each row corresponds to a frequency (number of individuals),
    with the corresponding count of SNPs that have that frequency.

    """
    nindiv = haplotype.shape[1]
    tmp_df = pd.DataFrame({"N_indiv": range(1, nindiv)})

    # getting unfolded sfs
    df_sfs = pd.DataFrame(allel.sfs(ac.T[1]), columns=["count_SNP"])
    df_sfs.index.name = "N_indiv"
    df_sfs.reset_index(inplace=True)
    df_sfs = df_sfs.merge(tmp_df, on="N_indiv", how="right").fillna(0).astype(int)

    return df_sfs


def process_file(file_name, path_to_SNP):
    '''
    Calculate the sfs and LD from one single SNP file
    :param file_name: file name of a SNP file .npz
    :param path_to_SNP: the folder containing the file
    :return: scenario, rep, afs, ld
    '''
    try:
        numbers = extract_numbers(file_name)
        if numbers is None:
            return
        scenario, rep = numbers
        SNP_file = np.load(os.path.join(path_to_SNP, file_name))
        snp = SNP_file['SNP']
        pos = SNP_file['POS']

        if any(np.diff(pos) < 0):
            pos = np.cumsum(pos)
        if pos.max() <= 1:
            pos = (pos * 2e6).round().astype(int)

        haplotype = allel.HaplotypeArray(snp.T)
        allel_count = haplotype.count_alleles()

        afs = sfs(haplotype, allel_count)
        afs = afs.set_index('N_indiv')
        afs['scenario'] = scenario

        ld = LD(haplotype, pos, circular=False, size_chr=2e6)
        ld["scenario"] = scenario
        ld = ld.drop(columns=['sem_r2'])

        return scenario, rep, afs, ld
    except Exception as e:
        logging.error("Error processing file {}: {}".format(file_name, e))
        return


def sum_stat(data_path, path_to_SNP, scenario_rep, max_cores=None):
    '''
    Compute the mean sum stat from each scenario, and store them in proper format
    :param data_path: folder that stores related data
    :param path_to_SNP: folder that stores SNP files
    :param max_cores: the maximal cores can be used for computing
    :return:
    '''

    # all .npz file name would be returned sorted.
    file_names = sorted(os.listdir(path_to_SNP),
                        key=lambda f: extract_numbers(f) if extract_numbers(f) else (float('inf'), float('inf')))
    if max_cores is None:
        max_cores = cpu_count()

    with Pool(processes=max_cores) as pool:
        results = pool.starmap(process_file, [(fn, path_to_SNP) for fn in file_names])
        results = [res for res in results if res is not None]

        scenario_data = {}
        for scenario, rep, afs, ld in results:
            if scenario not in scenario_data:
                scenario_data[scenario] = {"afs": [], "ld": []}

            scenario_data[scenario]["afs"].append(afs)
            scenario_data[scenario]["ld"].append(ld)

            # Check if this scenario has completed all the replications
            if len(scenario_data[scenario]["afs"]) == scenario_rep:
                mean_afs = pd.concat(scenario_data[scenario]["afs"]).groupby("N_indiv", observed=False).mean()
                mean_afs['scenario'] = scenario
                mean_afs.reset_index(inplace=True)
                mean_ld = pd.concat(scenario_data[scenario]["ld"]).groupby("dist_group", observed=False).mean()

                df_sfs = mean_afs.set_index('N_indiv')
                df_sfs_out = df_sfs.loc[df_sfs['scenario'] == scenario]
                df_sfs_out = df_sfs_out.drop(columns=['scenario'])
                df_sfs_out = df_sfs_out.stack(dropna=False)
                df_sfs_out.index = df_sfs_out.index.map('{0[1]}_{0[0]}'.format)
                df_sfs_out = df_sfs_out.to_frame().T
                df_sfs_out = df_sfs_out.set_index([[scenario]])

                df_ld_out = mean_ld.loc[np.array(mean_ld['scenario'] == scenario)]
                df_ld_out = df_ld_out.drop(columns=['scenario'])
                df_ld_out = df_ld_out.stack(dropna=False)
                df_ld_out.index = df_ld_out.index.map('{0[1]}_{0[0]}'.format)
                df_ld_out = df_ld_out.to_frame().T
                df_ld_out = df_ld_out.set_index([[scenario]])

                df = pd.merge(df_sfs_out, df_ld_out, left_index=True, right_index=True)
                col2drop = [col for col in df.columns if col.startswith('mean_dist_') or col.startswith('Count_')]
                df.drop(columns=col2drop, inplace=True)
                with open(os.path.join(data_path, 'preprocessed_data/sum_stat.csv'), 'a') as filename:
                    df.to_csv(filename, mode='a', header=filename.tell() == 0, index_label="scenario")

                # Clear the stored data for this scenario
                del scenario_data[scenario]

    print('Summary statistics stored')


if __name__ == "__main__":
    sum_stat(snakemake.params.datadir, os.path.join(snakemake.params.datadir, 'checked_SNPs'),
             max_cores=snakemake.params.max_cores, scenario_rep=snakemake.params.replication)
