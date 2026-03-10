import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re
import argparse
import itertools
import logging

from anospp_analysis.util import MOSQ_TARGETS, setup_logging, well_ordering, load_hap, load_comb_stats
from anospp_analysis.qc import plot_het_cov

def prep_mosquito_haps(hap_df, rc_threshold, rf_threshold):
    '''
    prepare mosquito haplotype dataframe
    remove plasmodium haplotypes
    change targets to integers
    returns haplotype dataframe
    '''

    logging.info('preparing mosquito haplotypes')

    hap_df = hap_df.astype({'target': str})
    filtered_hap_df = hap_df[
        (hap_df.reads >= rc_threshold) & (hap_df.reads_fraction >= rf_threshold)
        ]
    if filtered_hap_df.shape[0] < hap_df.shape[0]:
        logging.info(
            f'removed {hap_df.shape[0] - filtered_hap_df.shape[0]} haplotypes '
            f'with fewer than {rc_threshold} reads or fraction lower than {rf_threshold} of reads'
        )
    assert filtered_hap_df.shape[0] > 0, 'No haplotypes left after filtering, terminating'
    mosq_hap_df = filtered_hap_df[filtered_hap_df.target.isin(MOSQ_TARGETS)]
    mosq_hap_df = mosq_hap_df.astype({'target': int})

    #recompute reads coverage after filtering
    mosq_hap_df = recompute_haplotype_coverage(mosq_hap_df)

    return mosq_hap_df

def prep_reference_index(reference_path):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index from {reference_path}')

    reference_path = reference_path.rstrip('/')

    reference_version = reference_path.split('/')[-1]

    assert re.match(r'^nnv\d', reference_version), f'{reference_version} not recognised as NN ref version'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/haplotypes.tsv'), \
        f'reference version {reference_version} at {reference_path} does not contain required haplotypes.tsv file'
    ref_hap_df = pd.read_csv(f'{reference_path}/haplotypes.tsv', sep='\t')

    assert os.path.isfile(f'{reference_path}/multiallelism.tsv'), \
        f'reference version {reference_version} at {reference_path} does not contain required multiallelism.tsv file'
    true_multi_targets = pd.read_csv(f'{reference_path}/multiallelism.tsv', sep='\t')

    if os.path.isfile(f'{reference_path}/version.txt'):
        with open(f'{reference_path}/version.txt', 'r') as fn:
            for line in fn:
                version_name = line.strip()
    else:
        logging.warning('No version.txt file present for reference version {reference_version} at {reference_path}')
        version_name = 'unknown'
    if version_name != reference_version:
        logging.warning(f'Reference directory name {reference_version} does not match version.txt: {version_name}')

    allele_freqs = dict()
    colors = dict()
    for level in ['coarse', 'int', 'fine']:
        assert os.path.isfile(f'{reference_path}/allele_freq_{level}.npy'), \
            f'reference version {reference_version} at {reference_path} does not contain required allele_freq_{level}.npy file'
        af = np.load(f'{reference_path}/allele_freq_{level}.npy')
        allele_freqs[level] = af

        assert os.path.isfile(f'{reference_path}/sgp_{level}.txt'), \
            f'reference version {reference_version} at {reference_path} does not contain required sgp_{level}.txt file'
        sgp = []
        with open(f'{reference_path}/sgp_{level}.txt', 'r') as fn:
            for line in fn:
                sgp.append(line.strip())

        ref_hap_df[f'{level}_sgp'] = pd.Categorical(ref_hap_df[f'{level}_sgp'], sgp, ordered=True)
        assert ~ref_hap_df[f'{level}_sgp'].isna().any(), \
            f'reference version {reference_version} at {reference_path} has inconsistent labels between haplotypes.tsv and sgp_{level}.txt'

        if not os.path.isfile(f'{reference_path}/colors_{level}.npy'):
            logging.warning('No colors defined for plotting.')
        else:
            clr = np.load(f'{reference_path}/colors_{level}.npy')
            colors[level] = clr
        
    return(ref_hap_df, allele_freqs, true_multi_targets, colors, version_name)

def construct_kmer_dict(k):
    '''
    construct a k-mer dict
    associating each unique k-mer of length k with a unique non-negative integer <4**k
    bases are written in capitals
    returns a dictionary
    '''
    labels = []
    for i in itertools.product('ACGT', repeat=k):
        labels.append(''.join(i))
    kmerdict = dict(zip(labels, np.arange(4**k)))
    return kmerdict   

def parse_seqid(seqid):
    '''
    Parse seqid passed as a string 
    '''
    split_seqid = seqid.split('-')

    assert split_seqid[0] in MOSQ_TARGETS, f'seqid {seqid} refers to a non-mosquito target'
    try:
        parsed_seqid = (int(split_seqid[0]), int(split_seqid[1]))
    except:
        raise Exception(f'seqid {seqid} cannot be converted to integers')
    return parsed_seqid

def parse_seqids_series(seqids):
    '''
    Parse seqid or seqids passed as a pandas Series
    '''
    parsed_seqids = seqids.str.split('-', expand=True)
    # assert parsed_seqids.shape[1] == 2, 'Failed to parse seqids, we expected only one "-" in each seqid'
    parsed_seqids.columns = ['target', 'uidx']
    
    assert parsed_seqids['target'].isin(MOSQ_TARGETS).all(), 'Dataframe contains seqids referring to non-mosquito targets'
    try:
        parsed_seqids = parsed_seqids.astype(int)
    except:
        raise Exception('Dataframe contains seqids which cannot be converted to integers')
    return parsed_seqids

def construct_unique_kmer_table(mosq_hap_df, k, source):
    '''
    constructs a k-mer table of dimensions n_amp * maxallele * 4**k
    represting the k-mer table of each unique sequence in the dataframe
    maxallele is the maximum number of unique sequences per target
    n_amp is the number of mosquito targets
    input: k=k (length of k-mers), hap_df=dataframe with haplotypes
    output: k-mer table representing each unique haplotype in the hap dataframe
    '''

    logging.info(f'translating unique sequences from {source} to {k}-mers')

    kmerdict = construct_kmer_dict(k)
    #subset to unique haplotypes
    uniqueseq = mosq_hap_df[['seqid', 'consensus']].drop_duplicates()
    #determine shape of table by top seqid
    parsed_seqids = parse_seqids_series(uniqueseq.seqid)
    maxid = parsed_seqids['uidx'].max() + 1

    #initiate table to store kmer counts
    kmer_table = np.zeros((len(MOSQ_TARGETS), maxid, 4**k), dtype='int')
    #translate each unique haplotype to kmer counts
    for idx, seq in uniqueseq.iterrows():
        tgt = parsed_seqids.loc[idx, 'target']
        uid = parsed_seqids.loc[idx, 'uidx']
        consensus = seq.consensus
        for i in np.arange(len(consensus) - (k - 1)):
            kmer_table[tgt, uid, kmerdict[consensus[i:i+k]]] += 1
    return kmer_table

def identify_error_seqs(mosq_hap_df, kmers, k, n_error_snps):
    '''
    Identify haplotypes resulting from sequencing/PCR errors
    Cannot distinguish between true heterozygote, contaminated homozygote and homozygote with error sequence
    So only look for errors for unique sequences at multiallelic targets
    '''

    logging.info('identifying haplotypes resulting from sequencing/PCR errors')
    #set the k-mer threshold for the number of snps allowed for errors
    threshold = n_error_snps * k + 1
    seqid_size = mosq_hap_df.groupby('seqid').size()
    singleton_seqids = seqid_size[seqid_size == 1].index # used in query
    error_candidates = mosq_hap_df.query('(seqid in @singleton_seqids) & (nalleles > 2)')

    error_seqs = []
    for _, cand in error_candidates.iterrows():
        possible_sources = mosq_hap_df.query(
            '(sample_id == @cand.sample_id) & (target == @cand.target) & (seqid != @cand.seqid)')
        cand_parsed_seqid = parse_seqid(cand.seqid)
        possible_sources_parsed_seqids = parse_seqids_series(possible_sources.seqid)
        for possible_source in possible_sources_parsed_seqids['uidx']:
            abs_kmer_dist = np.abs(
                kmers[cand.target, cand_parsed_seqid[1], :] - kmers[cand.target, possible_source, :]
                ).sum()
            if abs_kmer_dist < threshold:
                error_seqs.append(cand.seqid)
                break
    
    logging.info(f'identified {len(error_seqs)} error sequences')

    return error_seqs

def recompute_haplotype_coverage(hap_df):
    hap_df = hap_df.drop(['total_reads', 'reads_fraction', 'nalleles'], axis=1)

    hap_df['total_reads'] = hap_df.groupby(by=['sample_id', 'target']) \
            ['reads'].transform('sum')

    hap_df['reads_fraction'] = hap_df['reads'] / hap_df['total_reads']

    hap_df['nalleles'] = hap_df.groupby(by=['sample_id', 'target']) \
            ['consensus'].transform('nunique')
    return hap_df

def compute_kmer_distance(kmers, ref_kmers, tgt, qidx, refidx):
    '''
    compute k-mer distance between query kmer count
    and ref kmer count(s)
    returns absolute and normalised distance
    '''
    #identify k-mer mismatches
    diff = np.abs(ref_kmers[tgt, refidx, :] - kmers[tgt, qidx, :])
    total = np.sum(ref_kmers[tgt, refidx, :] + kmers[tgt, qidx, :], axis=1)
    #sum to get absolute distance
    dist = np.sum(diff, axis=1)
    #normalise
    norm_dist = dist / total

    return dist, norm_dist

def find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers):
    '''
    identify the nearest neighbours of the unique haplotypes in the reference dataset 
    '''
    #get idxs occupied for each target
    parsed_ref_seqids = parse_seqids_series(ref_hap_df.seqid.drop_duplicates())
    ref_idxs_per_target = parsed_ref_seqids.groupby('target')['uidx'].unique()

    nndict = dict()

    logging.info(f'identifying nearest neighbours for {non_error_hap_df.seqid.nunique()} unique haplotypes')
    
    #loop through unique haplotypes
    unique_seqids = non_error_hap_df.seqid.unique()
    for seqid in unique_seqids:
        tgt, qidx = parse_seqid(seqid)
        #compute distance between focal hap and all same target haps in ref index
        dist, norm_dist = compute_kmer_distance(kmers, ref_kmers, tgt, qidx, ref_idxs_per_target[tgt])
        #Find nearest neighbours
        nn_qidx = ref_idxs_per_target[tgt][norm_dist==norm_dist.min()]
        #include in dict
        nndict[seqid] = (nn_qidx, norm_dist.min())

    return nndict

def lookup_assignment_proportion(q_seqid, allele_frequencies, tgt, nndict, weight=1):

    #lookup nearest neighbour identifiers
    nnids = nndict[q_seqid][0]
    #lookup allele frequencies of nnids
    af_nn = allele_frequencies[tgt, nnids, :]
    #sum allele frequencies over nnids
    summed_af_nn = np.sum(af_nn, axis=0)
    #normalise proportion and weigth in number of alleles
    assignment_proportion = weight * summed_af_nn / np.sum(summed_af_nn)
    return assignment_proportion

def add_nn_to_haplotypes(non_error_hap_df, nndict, ref_hap_df):

    def get_unique_sgps(x, level, ref_spp):
        seqids = x.split(';')
        match_sgps = ref_spp.loc[seqids, f'{level}_sgp'].transform(set)
        sgps = set().union(*match_sgps)# - set([np.nan])
        if len(sgps) == 0:
            return ''
        return ';'.join(sgps)

    logging.info('adding NN annotations to non-error haplotypes')

    non_error_hap_df['ref_nn_dist'] = non_error_hap_df['seqid'].apply(lambda x: nndict[x][1])
    non_error_hap_df['ref_nn_seqids'] = non_error_hap_df['seqid'].apply(
        lambda x: ";".join([x.split('-')[0] + '-' + str(y) for y in nndict[x][0]])
    )

    ref_spp = ref_hap_df.groupby('seqid')[['fine_sgp','int_sgp','coarse_sgp']].agg(set)

    non_error_hap_df['ref_coarse_sgps'] = non_error_hap_df['ref_nn_seqids'].apply(lambda x: get_unique_sgps(x, 'coarse', ref_spp))
    non_error_hap_df['ref_int_sgps'] = non_error_hap_df['ref_nn_seqids'].apply(lambda x: get_unique_sgps(x, 'int', ref_spp))
    non_error_hap_df['ref_fine_sgps'] = non_error_hap_df['ref_nn_seqids'].apply(lambda x: get_unique_sgps(x, 'fine', ref_spp))
    
    return non_error_hap_df

def perform_nn_assignment_samples(hap_df, ref_hap_df, nndict, allele_freqs, normalisation):
    '''
    The main NN assignment function
    it outputs three dataframes containing the assignment proportions to each species-group for the three levels
    '''
    #get samples with at least 10 targets
    test_samples = hap_df \
        .groupby('sample_id') \
        .filter(lambda x: x['target'].nunique() >= 10)['sample_id'] \
        .unique()

    logging.info(f'performing NN assignment for {len(test_samples)} samples with >=10 mosquito targets')

    #set up data-output as numpy arrays (will be made into dataframes later)
    results = {
        'coarse': np.zeros((len(MOSQ_TARGETS), len(test_samples), allele_freqs['coarse'].shape[2])),
        'int': np.zeros((len(MOSQ_TARGETS), len(test_samples), allele_freqs['int'].shape[2])),
        'fine': np.zeros((len(MOSQ_TARGETS), len(test_samples), allele_freqs['fine'].shape[2]))
    }

    for i, sample in enumerate(test_samples):
        #Restrict to targets amplified in focal sample
        targets = hap_df.loc[hap_df.sample_id == sample, 'target'].unique()
        
        #Per amplified target
        for tgt in targets:
            #Identify the unique IDs of the focal sample's haplotypes at target t
            alleles = hap_df.loc[
                (hap_df.sample_id == sample) & (hap_df.target == tgt),
                ['seqid', 'reads_fraction']]
            #for each haplotype
            for _, allele in alleles.iterrows():
                #for each assignment level
                for level in ['coarse', 'int', 'fine']:
                    if normalisation == 'n_alleles':
                    #lookup assignment proportion
                        assignment_proportion = lookup_assignment_proportion(
                            allele.seqid,
                            allele_freqs[level],
                            tgt,
                            nndict,
                            1 / alleles.shape[0]
                            )
                    elif normalisation == 'reads_fraction':
                        assignment_proportion = lookup_assignment_proportion(
                            allele.seqid,
                            allele_freqs[level],
                            tgt,
                            nndict,
                            allele.reads_fraction
                        )
                    else:
                        logging.error('Not a valid allelism_normalisation method.')
                    #table[tgt,nsmp,:] += assignment_proportion
                    results[level][tgt, i, :] += assignment_proportion

    #print(f'shape of results arrays is {results_coarse.shape}, {results_int.shape} and {results_fine.shape}')
    results_dfs = dict()
    for level in ['coarse', 'int', 'fine']:
        #Average assignment results over amplified targets
        res = np.nansum(results[level], axis=0)/np.sum(np.nansum(results[level], axis=0), axis=1)[:,None]
        #Convert results to dataframes
        results_dfs[level] = pd.DataFrame(
            res, 
            index=test_samples, 
            columns=ref_hap_df[f'{level}_sgp'].cat.categories
        )  
    return results_dfs, test_samples

def recompute_sample_coverage(comb_stats_df, non_error_hap_df):
    '''
    recompute coverage stats after filtering and error removal
    '''
    logging.info('recompute coverage stats')
    comb_stats_df.set_index('sample_id', inplace=True)

    #recompute multiallelic calls after filtering and error removal
    comb_stats_df['multiallelic_mosq_targets'] = (
        non_error_hap_df.groupby('sample_id')['target'].value_counts() > 2
        ).groupby(level='sample_id').sum()

    #recompute read counts after filtering and error removal
    comb_stats_df['mosq_reads'] = non_error_hap_df.groupby('sample_id')['reads'].sum()

    #recompute targets recovered after filtering and error removal
    comb_stats_df['mosq_targets_recovered'] = non_error_hap_df.groupby('sample_id')['target'].nunique()

    for col in ['multiallelic_mosq_targets', 'mosq_reads', 'mosq_targets_recovered']:
        comb_stats_df[col] = comb_stats_df[col].fillna(0).astype(int)

    comb_stats_df.reset_index(inplace=True)

    return comb_stats_df

def generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, results_dfs,
                        true_multi_targets, nn_asgn_threshold):

    logging.info('generating NN calls from assignment info')

    #Account for filtering and error removal
    comb_stats_df = recompute_sample_coverage(comb_stats_df, non_error_hap_df)

    #Record whether NN assignment was performed
    comb_stats_df.loc[comb_stats_df.sample_id.isin(test_samples), 'nn_assignment'] = 'yes'
    comb_stats_df.loc[comb_stats_df.nn_assignment.isnull(), 'nn_assignment'] = 'no'

    #Generate assignment hard calls if the threshold is met
    for level in ['coarse', 'int', 'fine']:
        asgn_dict = dict(results_dfs[level] \
            .loc[(results_dfs[level] >= nn_asgn_threshold).any(axis=1)] \
            .apply(
                lambda row: results_dfs[level].columns[row >= nn_asgn_threshold][0],
                axis=1))
        comb_stats_df[f'nn_{level}'] = comb_stats_df.sample_id.map(asgn_dict)

    comb_stats_df['nn_species_call'] = None
    comb_stats_df['nn_call_method'] = None
    # NN hierarchical assignment by level
    for level in ['fine', 'int', 'coarse']:
        leveldict = dict(zip(comb_stats_df.sample_id, comb_stats_df[f'nn_{level}']))
        is_id_on_level = (comb_stats_df.nn_species_call.isnull() & ~comb_stats_df[f'nn_{level}'].isnull())
        comb_stats_df.loc[is_id_on_level, 'nn_call_method'] = f'NN_{level}'
        comb_stats_df.loc[is_id_on_level, 'nn_species_call'] = comb_stats_df.loc[
            comb_stats_df.nn_call_method == f'NN_{level}', 'sample_id'
            ].map(leveldict)
    
    #Rainbow samples
    is_rainbow = (comb_stats_df.nn_species_call.isnull() & (comb_stats_df.nn_assignment == 'yes'))
    comb_stats_df.loc[is_rainbow, 'nn_call_method'] = 'NN'
    comb_stats_df.loc[is_rainbow, 'nn_species_call'] = 'RAINBOW_SAMPLE'

    #Samples with too few targets
    is_not_id = (comb_stats_df.nn_assignment == 'no')
    comb_stats_df.loc[is_not_id, 'nn_call_method'] = 'TOO_FEW_TARGETS'
    comb_stats_df.loc[is_not_id, 'nn_species_call'] = 'TOO_FEW_TARGETS'

    assert not comb_stats_df.nn_species_call.isnull().any(), 'some samples not assigned'
    assert not comb_stats_df.nn_call_method.isnull().any(), 'some samples not assigned'

    return comb_stats_df

def generate_summary(comb_stats_df, version_name):

    summary = [
        f'Nearest Neighbour assignment using reference version {version_name}',
        f'On run containing {comb_stats_df.sample_id.nunique()} samples',
        f'{(comb_stats_df.nn_assignment=="no").sum()} samples with < 10 targets lack NN assignment',
        f'{(~comb_stats_df.nn_coarse.isnull()).sum()} samples are assigned at coarse level',
        f'to {comb_stats_df.nn_coarse.nunique()} different species groups',
        f'{(~comb_stats_df.nn_int.isnull()).sum()} samples are assigned at intermediate level',
        f'to {comb_stats_df.nn_int.nunique()} different species groups',
        f'{(~comb_stats_df.nn_fine.isnull()).sum()} samples are assigned at fine level',
        f'to {comb_stats_df.nn_fine.nunique()} different species groups',
        f'{comb_stats_df.loc[comb_stats_df.nn_call_method == "NN_int", "sample_id"].nunique()} '
        f'samples with sufficient coverage could not be assigned at fine level',
        f'{comb_stats_df.loc[comb_stats_df.nn_call_method == "NN_coarse", "sample_id"].nunique()} '
        f'samples with sufficient coverage could not be assigned at intermediate level',
    ]
    return '\n'.join(summary)

def prep_stats_for_plotting(comb_stats_df, locov_rc):

    

    return comb_stats_df

def plot_assignment_proportions(comb_stats_df, nn_level_result_df, level_label, level_colors, run_id, plasm_assignment_df, plasm_colors, args):
    
    logging.info(f'generating {level_label} level plots')

    assert nn_level_result_df.index.isin(comb_stats_df['sample_id']).all(), \
        f'NN {level_label} samples do not match comb stats'
    
    assert nn_level_result_df.index.is_unique, \
        f'NN {level_label} samples are not unique'

    #Generate bar plots at given assignment level
    #add samples with <10 targets
    nn_level_result_df = pd.concat([
        nn_level_result_df, pd.DataFrame(
            index=comb_stats_df.loc[
                ~comb_stats_df.sample_id.isin(nn_level_result_df.index), 'sample_id'
                ]
            )
        ]).fillna(0)

    #Get row and col info from well_id for ordering samples
    comb_stats_df['row_id'] = comb_stats_df.well_id.str[0]
    comb_stats_df['col_id'] = comb_stats_df.well_id.str[1:].astype(int)
    comb_stats_df['well_id'] = well_ordering(comb_stats_df['well_id'])
    # comb_stats_df.sort_values(by=['plate_id', 'col_id', 'well_id'], inplace=True)
    
    # ma bands rely on multiallelics being numeric
    comb_stats_df['multiallelic_mosq_targets'] = comb_stats_df['multiallelic_mosq_targets'].fillna(0).astype(int)
    # multiallelics bands
    comb_stats_df['ma_band'] = '0'
    comb_stats_df.loc[comb_stats_df['multiallelic_mosq_targets'] > 0, 'ma_band'] = '1-2'
    comb_stats_df.loc[comb_stats_df['multiallelic_mosq_targets'] > 2, 'ma_band'] = '3-4'
    comb_stats_df.loc[comb_stats_df['multiallelic_mosq_targets'] > 4, 'ma_band'] = '5+'

    # locov relies on mosq_reads being numeric
    comb_stats_df['mosq_reads'] = comb_stats_df['mosq_reads'].fillna(0).astype(int)
    # mark low coverage samples - this can only be done once
    comb_stats_df['mosq_targets_plot'] = comb_stats_df['mosq_targets_recovered'].astype(str)
    comb_stats_df.loc[
        comb_stats_df['mosq_reads'] < args.locov_rc,
        'mosq_targets_plot'
    ] = comb_stats_df['mosq_targets_plot'] + '*'

    # multiallelics color scheme - applied to top ticks 
    ma_colors = {
        '0':'#808080',
        '1-2':'#FF9900',
        '3-4':'#cc00FF',
        '5+':'#FF0000'
    }

    # plasm color scheme - applied to bottom ticks
    if plasm_assignment_df is not None and plasm_colors is not None:
        logging.info('using plasm predictions to colour sample labels')
        plasm_assignment_df['plasmodium_species'] = plasm_assignment_df['plasmodium_species'].fillna('')
        plasm_spp = plasm_assignment_df.set_index('sample_id')['plasmodium_species'].to_dict()
        assert set(plasm_spp.keys()) == set(comb_stats_df.sample_id), \
            'plasmodium assignment samples do not match nn samples'

        # named species in legend - remove genus name
        plasm_legend_colors = {sp[11:]:color for sp, color in plasm_colors.iloc[:6].to_dict().items()}
        # other species collapsed
        assert plasm_colors.iloc[6:].nunique() == 1, \
            'plasmodium species color scheme not matching nn plot expectation'
        plasm_legend_colors['other'] = plasm_colors['unknown']
        # mixed/uninfected inferred during plotting
        plasm_legend_colors['mixed'] = '#000000'
        plasm_legend_colors['none'] = '#808080'

    # plot
    plates = comb_stats_df.plate_id.unique()
    nplates = comb_stats_df.plate_id.nunique()
    fig, axs = plt.subplots(nplates, 1, figsize=(20, 4 * nplates + 1))
    if nplates == 1:
        axs = [axs]
    for plate, ax in zip(plates, axs):
        plot_df = comb_stats_df[comb_stats_df.plate_id == plate].copy().sort_values('well_id').reset_index()
        plot_samples = plot_df['sample_id'] # TODO does not work for duplicated sample IDs
        # plot nn proportions for plate
        nn_level_result_plot_df = nn_level_result_df.loc[plot_samples]
        # TODO remove all-zero columns to speed up plotting - color index->label
        nn_level_result_plot_df.plot(
                kind='bar', stacked=True, width=1, ax=ax, color=level_colors
                )
        # remove zero size patches post-hoc
        for patch in ax.patches[:]:  # Copy list of patches to avoid modifying while iterating
            if isinstance(patch, mpatches.Rectangle):
                if patch.get_width() == 0 or patch.get_height() == 0:
                    patch.remove()
        ax.axhline(args.nn_assignment_threshold, color='k', ls=':', linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(plate, fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        
        # bottom ticks - species name coloured by plasm
        ax.set_xticks(range(plot_df.shape[0]))
        ax.set_xticklabels(plot_df['sample_name'])
        if plasm_assignment_df is not None and plasm_colors is not None:
            for i, r in plot_df.iterrows():
                sample_plasm_sp = plasm_spp[r.sample_id]
                # multiple species infection
                if len(sample_plasm_sp.split(';')) > 1:
                    ax.get_xticklabels()[i].set_color('black')
                # species in index
                elif sample_plasm_sp in plasm_colors.keys():
                    ax.get_xticklabels()[i].set_color(plasm_colors[sample_plasm_sp])
                # no infection
                else:
                    ax.get_xticklabels()[i].set_color('grey')
        ax.tick_params(axis='x', rotation=90)
        
        # top ticks - targets coloured by multiallelics band
        ax2 = ax.twiny()
        ax2.set_xticks(range(plot_df.shape[0]))
        ax2.set_xticklabels(plot_df['mosq_targets_plot'])
        for i, r in plot_df.iterrows():
            ax2.get_xticklabels()[i].set_color(ma_colors[r.ma_band])
        ax2.tick_params(axis='x', rotation=90)
        ax2.set_xlim(ax.get_xlim())
    plt.tight_layout()

    # add legends after adjusting layout so that they span multiple subplots
    ma_artist = axs[0].legend(
        handles=[mpatches.Patch(color=color, label=label) for label, color in ma_colors.items()],
        title=f'Total target count (top number) colored\nby multiallelic targets (* locov, <{args.locov_rc} reads)',
        alignment='left',
        loc='upper left',
        bbox_to_anchor=(1,1.1),
        fontsize=10,
        ncols=2
    )
    axs[0].add_artist(ma_artist)

    if plasm_assignment_df is not None and plasm_colors is not None:
        plasm_artist = axs[0].legend(
            handles=[mpatches.Patch(color=color, label=label) for label, color in plasm_legend_colors.items()],
            title='Specimen ID label (bottom) colored by\nPlasmodium species detected',
            alignment='left',
            loc='upper left',
            bbox_to_anchor=(1,0.725),
            fontsize=10,
            ncols=2
        )
        axs[0].add_artist(plasm_artist)

    # subset legend to values observed over the cutoff
    if args.legend_cutoff > 0:
        logging.info(f'subsetting legend to observed labels at min frequency {args.legend_cutoff}')
    flt_handles = []
    flt_labels = []
    for handle, label in zip(handles, labels):
        max_freq = nn_level_result_df[label].max()
        if max_freq >= args.legend_cutoff:
            flt_handles.append(handle)
            flt_labels.append(label)
    # consistent legend title
    leg_title_labels = {
        'coarse':'Coarse',
        'int':'Intermediate',
        'fine':'Fine'
    }
    leg_title = f'{leg_title_labels[level_label]} NN level mosquito species assignments'
    if args.legend_cutoff > 0:
        leg_title += f'\nwith observed proportion over {args.legend_cutoff}'
    # reverse species legend order to match barplot order
    axs[0].legend(
        flt_handles[::-1], flt_labels[::-1], 
        title=leg_title,
        alignment='left',
        loc='upper left',
        bbox_to_anchor=(1,0.1), 
        fontsize=10
        )
    # adding title in post - handling margins by savefig's bbox_inches='tight' at this point
    axs[0].set_title(f'NN assignment {level_label} level for {run_id}', fontsize=20)

    return fig, axs


def nn(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP NN data import started')

    hap_df = load_hap(args.haplotypes)
    run_id, comb_stats_df = load_comb_stats(args.stats)
    
    logging.info(f'starting NN assignment for {comb_stats_df.sample_id.nunique()} samples in run {run_id}')
    mosq_hap_df = prep_mosquito_haps(
        hap_df,
        args.hap_read_count_threshold,
        args.hap_reads_fraction_threshold
        )

    ref_hap_df, allele_freqs, true_multi_targets, colors, version_name = prep_reference_index(
        args.reference_path
        )
        
    nn_hap_fn = f'{args.outdir}/nn_hap_summary.tsv'
    nndict_fn = f'{args.outdir}/nn_dist_to_ref.tsv'
    if args.resume and os.path.isfile(nndict_fn) and os.path.isfile(nn_hap_fn):
        logging.warning(f'reading nndict from {nndict_fn}')
        nndict = {}
        with open(nndict_fn) as f:
            next(f)
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) == 3:
                    nndict[fields[0]] = ([int(i) for i in fields[1].split('|')], float(fields[2]))
        
        logging.warning(f'reading annotated haplotype data from {nn_hap_fn}')
        non_error_hap_df = pd.read_csv(nn_hap_fn, sep='\t')
    else:
        kmers = construct_unique_kmer_table(mosq_hap_df, args.kmer_length, source='samples')
        ref_kmers = construct_unique_kmer_table(ref_hap_df, args.kmer_length, source=version_name)

        error_seqs = identify_error_seqs(mosq_hap_df, kmers, args.kmer_length, args.n_error_snps)
        non_error_hap_df = mosq_hap_df[~mosq_hap_df.seqid.isin(error_seqs)]
        non_error_hap_df = recompute_haplotype_coverage(non_error_hap_df)
        
        nndict = find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers)
        nn_df = pd.DataFrame.from_dict(nndict, orient='index', columns=['nn_id_array', 'nn_dist'])
        nn_df['nn_id'] = ['|'.join(map(str, nn_id)) for nn_id in nn_df.nn_id_array]
        nn_df.index.name = 'seqid'
        nn_df[['nn_id', 'nn_dist']].to_csv(
            nndict_fn, 
            sep='\t',
            index=True
            )
        non_error_hap_df = add_nn_to_haplotypes(non_error_hap_df, nndict, ref_hap_df)
        logging.info(f'writing annotated haplotypes to {nn_hap_fn}')
        non_error_hap_df.to_csv(nn_hap_fn, index=False, sep='\t')

    nn_assignment_fn = f'{args.outdir}/nn_assignment.tsv'
    if args.resume and os.path.isfile(nn_assignment_fn):
        logging.warning(f'reading nn assignments from {nn_assignment_fn}')
        nn_stats_df = pd.read_csv(nn_assignment_fn, sep='\t')
        comb_stats_df = pd.merge(comb_stats_df, nn_stats_df, on='sample_id', how='left')
        results_dfs = {}
        for level in ['coarse', 'int', 'fine']:
            level_assignment_fn = f'{args.outdir}/assignment_{level}.tsv'
            logging.warning(f'reading {level} assignments from {level_assignment_fn}')
            # TODO fix heterogeneity in use of sample_id as index
            results_dfs[level] = pd.read_csv(level_assignment_fn, sep='\t', index_col=0)
    else:
        results_dfs, test_samples = perform_nn_assignment_samples(
            non_error_hap_df, 
            ref_hap_df, 
            nndict, 
            allele_freqs, 
            args.allelism_normalisation
        )
        for level in ['coarse', 'int', 'fine']:
            results_dfs[level].to_csv(f'{args.outdir}/assignment_{level}.tsv', sep='\t')

        comb_stats_df = generate_hard_calls(
            comb_stats_df, 
            non_error_hap_df, 
            test_samples,
            results_dfs, 
            true_multi_targets, 
            nn_asgn_threshold=args.nn_assignment_threshold
        )

        comb_stats_df['nn_ref'] = version_name
        logging.info(f'writing assignment results to {nn_assignment_fn}')
        comb_stats_df[[
            'sample_id',
            'run_id',
            'multiallelic_mosq_targets',
            'mosq_reads',
            'mosq_targets_recovered',
            'nn_assignment',
            'nn_coarse',
            'nn_int',
            'nn_fine',
            'nn_species_call',
            'nn_call_method',
            'nn_ref'
        ]].to_csv(nn_assignment_fn, index=False, sep='\t')
    
    summary_text = generate_summary(comb_stats_df, version_name)
    summary_fn = f'{args.outdir}/nn_summary.txt'
    logging.info(f'writing summary file to {summary_fn}')
    with open(summary_fn, 'w') as fn:
        fn.write(summary_text)

    if not args.no_plotting:

        fig, _, _, _ = plot_het_cov(non_error_hap_df, title='Mosquito filtered', run_id=run_id)
        fig.savefig(f'{args.outdir}/het_cov.png')

        if args.plasm_assignment is not None and args.plasm_colors is not None:
            plasm_df = pd.read_csv(args.plasm_assignment, sep='\t')
            plasm_colors = pd.read_csv(args.plasm_colors).set_index('species')['color']
        else:
            plasm_df = None
            plasm_colors = None
        
        for level in ['coarse', 'int', 'fine']:
            fig_fn = f'{args.outdir}/{level}_assignment.png'
            if args.resume and os.path.isfile(fig_fn):
                logging.warning(f'nn figure {fig_fn} exists, not re-genrating')
            else:
                fig, _ = plot_assignment_proportions(
                    comb_stats_df, 
                    results_dfs[level], 
                    level, 
                    colors[level],
                    run_id,
                    plasm_df,
                    plasm_colors,
                    args
                    )
                fig.savefig(fig_fn, bbox_inches='tight')

    logging.info('ANOSPP NN complete')

    
def main():
    
    parser = argparse.ArgumentParser('NN assignment for ANOSPP sequencing data')
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file generatd by prep', required=True)
    parser.add_argument('-s', '--stats', help='Comb stats tsv file generated by prep', required=True)
    parser.add_argument('-r', '--reference_path', 
                        help='Path to reference index directory, expected to end with e.g. nnv2 ',
                        required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: nn', default='nn')
    parser.add_argument('--no_plotting', help='Do not generate plots. Default: False',
                        action='store_true',
                        default=False)
    parser.add_argument('--allelism_normalisation',
                        help='Normalisation method over multiple alleles. Options: [n_alleles,reads_fraction]. '
                        'Default: n_alleles',
                        choices=['n_alleles', 'reads_fraction'],
                        default='n_alleles')
    parser.add_argument('--hap_read_count_threshold',
                        help='Minimum number of reads for supported haplotypes.  Default: 10',
                        default=10, type=int)
    parser.add_argument('--hap_reads_fraction_threshold',
                        help='Minimum fraction of reads for supported haplotypes. Default: 0.1',
                        default=0.1, type=float)
    parser.add_argument('--locov_rc',
                        help='Samples with fewer than this number of reads are marked as low coverage in plot. Default: 1000',
                        default=1000, type=int)
    parser.add_argument('--nn_assignment_threshold',
                        help='Required fraction for calling assignment. Default: 0.7 (changed in nnv2)',
                        default=0.7, type=float)
    parser.add_argument('--n_error_snps',
                        help='Maximum number of snps for a multi-allelic sequence to be considered a sequencing or PCR error. Default: 2',
                        default=2, type=int)
    parser.add_argument('-k', '--kmer_length',
                        help='Length of k-mers to use. Note that NNoVAE has been developed and tested for k=8, '
                        'so accuracy of results cannot be guaranteed with other values of k. Default: k=8',
                        default=8, type=int)
    parser.add_argument('--plasm_assignment',
                        help='Path to plasm_assignment.tsv file used for sample label colouring '
                        'in nn plots. Default: None - colouring not applied',
                        default=None)
    parser.add_argument('--plasm_colors',
                        help='Path to species_colours.csv from plasm reference directory '
                        'used for sample label colouring in nn plots. '
                        'Default: None - colouring not applied',
                        default=None)
    parser.add_argument('--legend_cutoff',
                        help='Minimum observed NN assignment proportion '
                        'for species label to be added to the legend '
                        'Default: 0 - include all species in reference index',
                        default=0, type=float)
    parser.add_argument('--resume',
                        help='Do not re-generate nn_dist_to_ref.tsv and nn_assignment.tsv '
                        'if those are present in the output directory',
                        action='store_true', default=False)
    parser.add_argument('-v', '--verbose',
                        help='Include INFO level log messages',
                        action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    nn(args)


if __name__ == '__main__':
    main()