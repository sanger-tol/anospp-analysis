import pandas as pd
import argparse
import subprocess
import glob
import os
import logging

from anospp_analysis.util import seqid_generator, setup_logging, well_id_mapper, lims_well_id_mapper, load_hap, CUTADAPT_TARGETS, MOSQ_TARGETS, PLASM_TARGETS

def parse_asv_tsv(asv_table):

    logging.info(f'preparing ASV table from {asv_table}')

    rows = []

    with open(asv_table) as f:
        header = f.readline().rstrip("\n").split("\t")
        
        # structure: ASV_ID | sample_1 ... sample_n | sequence
        sample_ids = header[1:-1]

        for line in f:
            parts = line.rstrip("\n").split("\t")
            asv_id = parts[0]
            sequence = parts[-1]
            counts = parts[1:-1]

            for sample_id, val in zip(sample_ids, counts):
                if val != "0":  # fast string check, avoids int conversion
                    rows.append((asv_id, sample_id, sequence, int(val)))

    logging.info(f'found {len(rows)} unique sequences')

    return pd.DataFrame(rows, columns=["asv_id", "sample_id", "sequence", "reads"])

def reverse_complement(seq):

    ambiguous_dna_complement = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "M": "K",
        "R": "Y",
        "W": "W",
        "S": "S",
        "Y": "R",
        "K": "M",
        "V": "B",
        "H": "D",
        "D": "H",
        "B": "V",
        "X": "X",
        "N": "N",
    }

    seq = seq.upper()

    rc_seq = ''
    for nt in seq[::-1]:
        rc_seq += ambiguous_dna_complement[nt]

    return rc_seq

def seq_to_fasta(asv_df, fasta_fn, rc=False):

    extra_msg = ' reverse complement' if rc else ''
    
    logging.info(f'writing{extra_msg} ASV sequences to fasta file {fasta_fn}')

    seq_df = asv_df[[
        'asv_id',
        'sequence'
    ]].drop_duplicates().set_index('asv_id')

    with open(fasta_fn, 'w') as outfile:
        for seqid, seq in seq_df['sequence'].items():
            if rc:
                rc_seq = reverse_complement(seq)
                outfile.write(f'>{seqid}_rc\n{rc_seq}\n')
            else:
                outfile.write(f'>{seqid}\n{seq}\n')

def run_cutadapt(fasta, primers, cutadapt_args, work_dir):

    cmd = f"cutadapt {cutadapt_args} -g file:{primers} "
    cmd += f"-o {work_dir}/ASV_{{name}}.fa {fasta}"

    logging.info(cmd)

    process = subprocess.run(cmd.split(), capture_output=True, text=True)
    process.check_returncode()

def get_deplex_df(work_dir):
    '''
    Read demultiplexed fasta into dataframe with
    seqid: target, trimmed_sequence
    '''
    
    logging.info(f'parsing deplexed sequences from {work_dir}')

    deplex_dict = dict()
    # iterate over deplexed fasta files
    for fa in sorted(glob.glob(f'{work_dir}/ASV_*.fa')):
        target = fa.split('/')[-1].split('.')[0].split('_', maxsplit=1)[1]
        # basic parser
        with open(fa) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    seqid = line[1:]
                    if seqid in deplex_dict:
                        logging.warning(f'duplicate seqid {seqid} found in deplexed fasta files')
                    deplex_dict[seqid] = {'target':target,'trimmed_sequence':''}
                else:
                    deplex_dict[seqid]['trimmed_sequence'] += line
        # proper parser
        # for record in SeqIO.parse(fa, format='fasta'):
        #     deplex_dict[record.name] = {'target':target,'trimmed_sequence':str(record.seq)}
    deplex_df = pd.DataFrame.from_dict(deplex_dict, orient="index")

    return deplex_df

def cutadapt_deplex(asv_df, primers, cutadapt_args, work_dir='work', rc=False):

    logging.info(f'cutadapt deplexing started at {work_dir}')

    os.makedirs(work_dir, exist_ok=True)

    fasta_fn = f'{work_dir}/input_seqs.fasta'
    seq_to_fasta(asv_df, fasta_fn, rc=False)

    run_cutadapt(fasta_fn, primers, cutadapt_args, work_dir)

    if rc:
        os.makedirs(f'{work_dir}/rc', exist_ok=True)
        fasta_fn_rc = f'{work_dir}/rc/input_seqs_rc.fasta'
        seq_to_fasta(asv_df, fasta_fn_rc, rc=True)
        run_cutadapt(fasta_fn_rc, primers, cutadapt_args, f'{work_dir}/rc')
    
    logging.info('combining dada output with deplexing info')

    deplex_df = get_deplex_df(work_dir)
    if rc:
        logging.info('adding reverse complement sequences')
        rc_deplex_df = get_deplex_df(work_dir + '/rc')
        assert deplex_df.shape[0] == rc_deplex_df.shape[0], \
            'mismatch in deplex and reverse complement sequences number'
        n_rc = 0
        for seqid, r in deplex_df.iterrows():
            rc_seqid = f'{seqid}_rc'
            rc_r = rc_deplex_df.loc[rc_seqid]
            if (r.target == 'unknown') and (rc_r.target != 'unknown'):
                n_rc += 1
                for col in ('trimmed_sequence', 'target'):
                    deplex_df.loc[seqid, col] = rc_r[col]
        logging.info(f'added {n_rc} reverse complement sequences matching targets')

    return deplex_df

def prep_hap_df(asv_df, deplex_df):

    hap_df = asv_df.merge(
        deplex_df, 
        how='inner',
        left_on='asv_id', 
        right_index=True,
        validate='many_to_one'
    )

    
    if not hap_df['target'].isin(CUTADAPT_TARGETS).all():
        logging.warning('non-ANOSPP targets detected in deplexing')
    hap_df.rename(
        columns={
            'sequence':'untrimmed_sequence',
            'trimmed_sequence':'consensus'
            },
        inplace=True)
    # remove unsupported sequences
    hap_df = hap_df.query('reads > 0').copy()
    # collapse identical trimmed sequences
    pre_collapse_nseq = len(hap_df)
    hap_df = hap_df.groupby(['sample_id', 'target', 'consensus'])['reads'].sum().reset_index()
    post_collapse_nseq = len(hap_df)
    if post_collapse_nseq != pre_collapse_nseq:
        logging.info(
            f'collapsed {pre_collapse_nseq - post_collapse_nseq} sequences with identical inserts'
            )

    logging.info('further annotating haps')
    # duplicated in util.load_hap, but avoids re-compute
    hap_df['total_reads'] = hap_df \
        .groupby(by=['sample_id', 'target']) \
        ['reads'].transform('sum')

    hap_df['reads_fraction'] = hap_df['reads'] / hap_df['total_reads']

    hap_df['nalleles'] = hap_df \
        .groupby(by=['sample_id', 'target']) \
        ['consensus'].transform('nunique')

    hap_df = seqid_generator(hap_df)

    return hap_df

def prep_samples(samples_fn, run_id=None):
    '''
    Prepare sample manifest used for anospp pipeline
    '''

    logging.info(f'preparing sample manifest from {samples_fn}')
    # allow reading from tsv (new style) or csv (old style)
    if samples_fn.endswith('csv'):
        logging.warning('csv manifest detected, assuming legacy format, converting to new conventions')
        samples_df = pd.read_csv(samples_fn, sep=',', dtype='str')
        samples_df.rename(columns=({
            'Source_sample':'sample_id',
            'Run':'run_id',
            'Lane':'lane_index',
            'Tag':'tag_index',
            'Replicate':'replicate_id'
            }), 
            inplace=True)
    elif samples_fn.endswith('tsv'):
        # logging.info(f'preparing sample manifest from new style file {samples_fn}')
        samples_df = pd.read_csv(samples_fn, sep='\t', dtype='str')
        if 'derived_sample_id' in samples_df.columns:
            logging.info('found derived_sample_id column, using it as sample_id')
            samples_df.rename(columns=({'derived_sample_id':'sample_id'}), inplace=True)
    else:
        raise ValueError(f'Expected {samples_fn} to be in either tsv (new) or csv (old) format')    

    logging.info('inferring run_id, lane_index and tag_index')
    if run_id is not None:
        logging.info(f'overriding run_id with provided value {run_id}')
        samples_df['run_id'] = run_id
    if 'irods_path' in samples_df.columns:
        logging.warning('inferring run_id, lane_index and tag_index from irods_path column')
        assert samples_df.irods_path.fillna('').str.match('/seq/\d{5}/\d{5}_\d#\d+.cram').all(), \
            ('tsv sample manifest input requires irods_path column to be present '
            'and match "/seq/12345/12345_1#123.cram"')
        samples_df['run_id'] = samples_df.irods_path.str.split('/').str.get(2)
        samples_df[['lane_index', 'tag_index']] = samples_df.irods_path \
            .str.split('/').str.get(3) \
            .str.split('_').str.get(1) \
            .str.split('.').str.get(0) \
            .str.split('#', expand=True)
    elif 'lane_index' in samples_df.columns and 'tag_index' in samples_df.columns:
        logging.info('found lane_index and tag_index columns, using them "as is"')
    else:
        logging.warning('no lane_index and tag_index columns found, inferring them from sample order')
        samples_df['lane_index'] = 1
        samples_df['tag_index'] = [i + 1 for i in range(len(samples_df))]

    for col in ('sample_id',
                'run_id',
                'lane_index',
                'tag_index'):
        assert col in samples_df.columns, f'samples column {col} not found'
    samples_df['run_id'] = samples_df['run_id'].astype(int)
    samples_df['lane_index'] = samples_df['lane_index'].astype(int)
    samples_df['tag_index'] = samples_df['tag_index'].astype(int)
    
    logging.info('inferring 96-well plate_id and well_id')
    if 'plate_id' in samples_df.columns and 'well_id' in samples_df.columns:
        logging.info('found plate_id and well_id columns, using them as is')
    else:
        # sample_id as `{plate_id}_{well_id}[-{sanger_sample_id}]` 
        try:
            plate_well_ids = samples_df['sample_id'].str.rsplit('-', n = 1).str.get(0)
            samples_df[['plate_id', 'well_id']] = plate_well_ids.str.rsplit('_', n = 1, expand=True)
            # do not allow for non-standard well IDs
            assert samples_df.well_id.isin(well_id_mapper().values()).all()
            # do not allow for duplicate plate IDs - issues with plotting
            assert (samples_df.plate_id.value_counts() <= 96).all()
            logging.info('inferring plate_id and well_id from sample_id')
        except:
            logging.warning('inferring plate_id and well_id from tags')
            samples_df['plate_id'] = samples_df.apply(lambda r: f'p_{r.run_id}_{(r.tag_index - 1) // 96 + 1}',
                axis=1)
            samples_df['well_id'] = (samples_df.tag_index % 96).replace(well_id_mapper())
    
    assert ~samples_df.plate_id.isna().any(), 'Could not infer plate_id for all samples'
    assert ~samples_df.well_id.isna().any(), 'Could not infer well_id for all samples'
    assert samples_df.well_id.isin(well_id_mapper().values()).all(), 'Found well_id outside A1...H12'
    
    logging.info('inferring 384-well lims_plate_id and lims_well_id')
    # id_library_lims as `{lims_plate_id}:{lims_well_id}`
    if ('id_library_lims' in samples_df.columns and
        samples_df.id_library_lims.str.contains(':').all()):
            logging.info('inferring lims_plate_id from id_library_lims')
            samples_df[['lims_plate_id', 'lims_well_id']] = samples_df.id_library_lims.str.split(
                ':', n = 1, expand=True
                )
    else:
        logging.info('inferring lims_plate_id from tags')
        samples_df['lims_plate_id'] = samples_df.apply(
            lambda r: f'lp_{r.run_id}_{(r.name) // 384 + 1}',
            axis=1
            )
        samples_df['lims_well_id'] = (samples_df.tag_index % 384).replace(lims_well_id_mapper())
    assert ~samples_df.lims_plate_id.isna().any(), 'Could not infer plate_id for all samples'
    assert ~samples_df.lims_well_id.isna().any(), 'Could not infer well_id for all samples'
    assert samples_df.lims_well_id.isin(lims_well_id_mapper().values()).all(), 'Found well_id outside A1...H12'
    
    logging.info('inferring short sample_name for plotting - part before last dash in sample_id')
    samples_df['sample_name'] = samples_df['sample_id'].str.rsplit('-', n=1).str.get(0)

    if 'upstream_id' in samples_df.columns:
        logging.info('found upstream_id column for mapping in ASV and stats tables')
        assert samples_df.upstream_id.is_unique, 'upstream_id is not unique in samples table'
    else:
        logging.info('no upstream_id column found, expecting sample_id in ASV and stats tables')

    run_id = samples_df['run_id'].iloc[0]
    logging.info(f'first run record assumed to be the run ID for plot titles: {run_id}')

    return run_id, samples_df

def prep_stats(stats_fn, sample_df):
    '''
    load DADA2 stats table from either DADA2_stats.tsv
    or overall_summary.txt file of ampliseq pipeline
    
    For legacy stats table, summarise across targets
    '''

    logging.info(f'preparing DADA2 statistics from {stats_fn}')

    stats_df = pd.read_csv(stats_fn, sep='\t')

    stats_df.rename(columns={
        # compatibility with legacy format
        's_Sample':'sample_id',
        'final':'dada2_postfilter_reads',
        # compatibility with new format 
        'sample':'sample_id',
        # generic renaming for DADA2 stats
        'DADA2_input':'dada2_input_reads',
        'filtered':'dada2_filtered_reads',
        'denoised':'dada2_denoised_reads',
        'denoisedF':'dada2_denoisedf_reads',
        'denoisedR':'dada2_denoisedr_reads',
        'merged':'dada2_merged_reads',
        'nonchim':'dada2_nonchim_reads'
        },
        inplace=True)
    
    assert 'sample_id' in stats_df.columns, 'stats column sample_id not found'

    if 'upstream_id' in sample_df.columns:
        logging.warning(
            'found upstream_id column in sample manifest, '
            'renaming samples in stats table to match')
        assert stats_df.sample_id.isin(sample_df.upstream_id).all(), \
            'found sample_id in stats table not matching sample manifest'
        stats_df['sample_id'] = stats_df['sample_id'].apply(lambda x: sample_df[sample_df.upstream_id == x].iloc[0]['sample_id'])

    # overall_summary.tsv values not compatible with stats model
    stats_df.drop(
            columns=['cutadapt_reverse_complemented', 'cutadapt_passing_filters_percent'],
            inplace=True, 
            errors='ignore'
            )

    for col in stats_df.columns:
        if col != 'sample_id':
            if stats_df[col].dtype == 'object' and stats_df[col].str.contains(',').any():
                stats_df[col] = stats_df[col].str.replace(',', '').fillna(0).astype(int)
            if stats_df[col].dtype != int:
                stats_df[col] = stats_df[col].fillna(0).astype(int)
            assert stats_df[col].dtype == int, f'stats column {col} expected to be integer {stats_df[col].head()}'

    is_dada_stats = stats_df.columns.str.startswith('dada2_').any()

    if not is_dada_stats:
        logging.warning('stats not in DADA2 format, using as is')
        return stats_df
    
    logging.info('found DADA2 stats columns, checking for consistency')
    # denoising happens for F and R reads independently, we take minimum of those 
    # as an estimate for denoised read count
    if 'dada2_denoisedf_reads' in stats_df.columns and 'dada2_denoisedr_reads' in stats_df.columns:
        logging.info('found DADA2 stats for paired end reads')
        stats_df['dada2_denoised_reads'] = stats_df[[
            'dada2_denoisedf_reads',
            'dada2_denoisedr_reads'
            ]].min(axis=1)
        assert 'dada2_merged_reads' in stats_df.columns, f'stats column dada2_merged_reads not found'
    else:
        logging.info('found DADA2 stats for single end reads')
        assert 'dada2_denoised_reads' in stats_df.columns, f'stats column dada2_denoised_reads not found'
        # fake merged - same as denoised
        stats_df['dada2_merged_reads'] = stats_df['dada2_denoised_reads']
    # legacy stats calculated separately for each target, merging
    if 'target' in stats_df.columns:
        logging.warning(f'summarising legacy DADA2 statistics across targets')
        stats_df = stats_df.groupby('sample_id').sum(numeric_only=True).reset_index()
    # overall_summary.txt
    if 'cutadapt_total_processed' in stats_df.columns:
        stats_df.rename(columns={
            'cutadapt_total_processed':'total_reads',
            'cutadapt_passing_filters':'readthrough_pass_reads'
        },
        inplace=True)
    # DADA2_stats
    else:
        logging.warning(
            'DADA2_stats.tsv provided instead of overall_summary.txt, '
            'cutadapt readthrough stats will be missing'
            )
        stats_df['total_reads'] = stats_df['dada2_input_reads']
        stats_df['readthrough_pass_reads'] = stats_df['dada2_input_reads']
    
    return stats_df

def combine_stats(stats_df, hap_df, samples_df):

    logging.info('preparing combined per-sample stats')

    assert set(stats_df.sample_id) - set(samples_df.sample_id) == set(), \
        'sample_id mismatch between samples and stats, QC results will be compromised'
    if set(samples_df.sample_id) - set(stats_df.sample_id) != set():
        logging.warning('some samples missing from stats, filling with zero reads')
    
    assert set(hap_df.sample_id) - set(samples_df.sample_id) == set(), \
        'sample_id mismatch between haps and samples, QC results will be compromised'
    if set(samples_df.sample_id) - set(hap_df.sample_id) != set():
        logging.warning('some samples missing from haps, assume all haps were lost in filtering')

    stats_cols = [col for col in stats_df.columns if col != 'sample_id']
    comb_stats_df = pd.merge(samples_df, stats_df, on='sample_id', how='left', validate='one_to_one')
    for col in stats_cols:
        comb_stats_df[col] = comb_stats_df[col].fillna(0).astype(int)
    comb_stats_df.set_index('sample_id', inplace=True)
    
    comb_stats_df['target_reads'] = hap_df[hap_df.target != 'unknown'] \
        .groupby('sample_id')['reads'].sum()
    comb_stats_df['target_reads'] = comb_stats_df['target_reads'].fillna(0).astype(int)

    comb_stats_df['overall_filter_rate'] = comb_stats_df['target_reads'] / comb_stats_df['total_reads']
    comb_stats_df['overall_filter_rate'] = comb_stats_df['overall_filter_rate'].fillna(0).round(3)

    comb_stats_df['unassigned_asvs'] = hap_df[hap_df.target == 'unknown'] \
        .groupby('sample_id')['consensus'].nunique()
    comb_stats_df['unassigned_asvs'] = comb_stats_df['unassigned_asvs'].fillna(0).astype(int)
    
    comb_stats_df['targets_recovered'] = hap_df[hap_df.target != 'unknown'] \
        .groupby('sample_id')['target'].nunique()
    comb_stats_df['targets_recovered'] = comb_stats_df['targets_recovered'].fillna(0).astype(int)
    
    if not hap_df.target.isin(CUTADAPT_TARGETS).any():
        logging.warning('no ANOSPP targets detected in haps, skipping target-specific stats')
    else:
        comb_stats_df['raw_mosq_targets_recovered'] = hap_df[hap_df.target.isin(MOSQ_TARGETS)] \
            .groupby('sample_id')['target'].nunique()
        comb_stats_df['raw_mosq_targets_recovered'] = comb_stats_df['raw_mosq_targets_recovered'].fillna(0).astype(int)

        comb_stats_df['raw_multiallelic_mosq_targets'] = (
            hap_df[hap_df.target.isin(MOSQ_TARGETS)].groupby('sample_id')['target'].value_counts() > 2
            ).groupby(level='sample_id').sum()
        comb_stats_df['raw_multiallelic_mosq_targets'] = comb_stats_df['raw_multiallelic_mosq_targets'].fillna(0).astype(int)
        
        comb_stats_df['raw_mosq_reads'] = hap_df[hap_df.target.isin(MOSQ_TARGETS)] \
            .groupby('sample_id')['reads'].sum()
        comb_stats_df['raw_mosq_reads'] = comb_stats_df['raw_mosq_reads'].fillna(0).astype(int)
        
        for pt in PLASM_TARGETS:
            ptl = pt.lower()
            comb_stats_df[f'{ptl}_reads'] = hap_df[hap_df.target == pt] \
                .groupby('sample_id')['reads'].sum()
            comb_stats_df[f'{ptl}_reads'] = comb_stats_df[f'{ptl}_reads'].fillna(0).astype(int)
    
    comb_stats_df.reset_index(inplace=True)
    comb_stats_df.sort_values(by=['lane_index','tag_index'], inplace=True)
        
    return comb_stats_df

def prep(args):

    setup_logging(verbose=args.verbose)
    
    logging.info('ANOSPP data prep started')

    os.makedirs(args.out_dir, exist_ok=True)

    run_id, samples_df = prep_samples(args.manifest, args.run_id)

    asv_df = parse_asv_tsv(args.asv_table)

    deplex_df = cutadapt_deplex(asv_df, args.primers, args.cutadapt_args, args.work_dir, rc=args.rc)

    hap_df = prep_hap_df(asv_df, deplex_df)

    hap_fn = f'{args.out_dir}/haps.tsv'

    # hap_df[[
    #     'sample_id',
    #     'target',
    #     'consensus',
    #     'reads',
    #     'total_reads',
    #     'reads_fraction',
    #     'nalleles',
    #     'seqid'
    # ]]
    hap_df.to_csv(hap_fn, sep='\t', index=False)

    hap_df = load_hap(hap_fn)
    
    if args.stats is None:
        logging.warning('no DADA2 stats file provided, using ASV table read counts as total reads')
        stats_df = asv_df.drop(columns=['sequence']).sum().reset_index().rename(columns={
            'index':'sample_id',
            0:'total_reads'
        })
    else:
        stats_df = prep_stats(args.stats, samples_df)

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)

    comb_stats_fn = f'{args.out_dir}/comb_stats.tsv'

    comb_stats_df.to_csv(comb_stats_fn, sep='\t', index=False)

    logging.info('ANOSPP data prep complete')

def main():
    
    parser = argparse.ArgumentParser("Convert DADA2 output to ANOSPP haplotypes tsv")
    parser.add_argument('-a', '--asv_table', 
                        help=('table of read counts, rows ASVs, columns samples and sequence -'
                              'for ampliseq, DADA2_table.tsv'),
                        required=True)
    parser.add_argument('-p', '--primers', 
                        help='fasta file for target deplexing with cutadapt', 
                        required=True)
    parser.add_argument('-m', '--manifest', 
                        help='samples manifest tsv file, used to infer sample mapping and plate layouts', 
                        required=True)
    parser.add_argument('-s', '--stats', 
                        help=('read filtering stats overall_summary.tsv - if not provided, '
                              'counts from ASV table used as total reads'), 
                        default=None)
    parser.add_argument('-o', '--out_dir', 
                        help='output directory for haplotypes and stats tsv files. Default: prep', 
                        default='prep')
    parser.add_argument('-w', '--work_dir', 
                        help='working directory for intermediate files. Default: work',
                        default='work')
    parser.add_argument('-i', '--run_id',
                        help=('use this run_id and sample order to infer run, lane and tag instead of `irods_path` column.' 
                              'Default: do not override'),
                        default=None)
    parser.add_argument('-r', '--rc', 
                        help=('also run deplexing on reverse complement ASVs, '
                              'useful when amplicon orientation is not defined by barcodes'), 
                        action='store_true')
    parser.add_argument('-c', '--cutadapt_args',
                        help=('Additional cutadapt arguments applied at target deplexing. ' 
                              'Default: "-O 10 --match-read-wildcards"'),
                        default='-O 10 --match-read-wildcards')
    parser.add_argument('-v', '--verbose', 
                        help='include INFO level log messages', action='store_true')

    args = parser.parse_args()

    args.work_dir=args.work_dir.rstrip('/')
    args.out_dir=args.out_dir.rstrip('/')
    for fn in args.asv_table, args.primers, args.manifest:
        assert os.path.isfile(fn), f'{fn} not found'

    prep(args)

if __name__ == '__main__':
    main()