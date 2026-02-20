import pandas as pd
import argparse

from anospp_analysis.util import *

def validate_aggregation(comb_df):

    logging.info('checking data types')

    for col in [
        'sample_id', #'irods_path', 'id_library_lims', 'id_study_lims', 'sanger_sample_id', 'front_barcode', 'rear_barcode'
        'run_id', 'lane_index', 'tag_index', 'plate_id',
        'well_id', 'lims_plate_id', 'lims_well_id', 'sample_name',
        'total_reads', 'readthrough_pass_reads', 'dada2_input_reads',
        'dada2_filtered_reads', 'dada2_denoised_reads', 'dada2_merged_reads',
        'dada2_nonchim_reads', 'target_reads', 'overall_filter_rate',
        'unassigned_asvs', 'targets_recovered', 'raw_mosq_targets_recovered',
        'raw_multiallelic_mosq_targets', 'raw_mosq_reads', 'p1_reads', 'p2_reads', 
        'p1_reads_pass', 'p2_reads_pass', 'plasmodium_detection_group', 'plasmodium_detection_species',
        'plasm_ref',
        # skip plasm hap
        'multiallelic_mosq_targets', 'mosq_reads', 'mosq_targets_recovered',
        'nn_assignment', 'nn_species_call', 'nn_call_method', 'nn_ref',
        # skip vae
        'nnovae_mosquito_species', 'nnovae_call_method'
        ]:
        assert ~comb_df[col].isna().any(), f'missing {col} values found'

    for col in [
        'run_id', 'lane_index', 'tag_index',
        'total_reads', 'readthrough_pass_reads', 'dada2_input_reads',
        'dada2_filtered_reads', 'dada2_denoised_reads', 'dada2_merged_reads',
        'dada2_nonchim_reads', 'target_reads',
        'unassigned_asvs', 'targets_recovered', 'raw_mosq_targets_recovered',
        'raw_multiallelic_mosq_targets', 'raw_mosq_reads', 'p1_reads', 'p2_reads', 
        'p1_reads_pass', 'p2_reads_pass',
        'multiallelic_mosq_targets', 'mosq_reads', 'mosq_targets_recovered'
        ]:
        assert pd.api.types.is_integer_dtype(comb_df[col]), f'{col} datatype is not integer'
        assert (comb_df[col] >= 0).all(), f'{col} contains negative values'

    for col in ['overall_filter_rate']:
        assert pd.api.types.is_numeric_dtype(comb_df[col]), f'{col} datatype is not numeric'

    # for col in ['mean1', 'mean2', 'mean3', 'sd1', 'sd2', 'sd3']:
    #     assert pd.api.types.is_numeric_dtype(comb_df[col]) or comb_df[col].empty, f'{col} datatype is not numeric'

    logging.info('checking columns contents')

    assert comb_df.sample_id.is_unique, 'duplicated sample_id found'

    assert len(comb_df.run_id.unique()) == 1, 'more than a single run_id found'

    assert comb_df.well_id.isin(well_id_mapper().values()).all(), 'non A1...H12 well_id found'

    assert comb_df.lims_well_id.isin(lims_well_id_mapper().values()).all(), 'non A1...P24 lims_well_id found'

    for (colp, coln) in [
        ('total_reads', 'readthrough_pass_reads'),
        ('readthrough_pass_reads', 'dada2_input_reads'),
        ('dada2_input_reads', 'dada2_filtered_reads'),
        ('dada2_filtered_reads', 'dada2_denoised_reads'),
        ('dada2_denoised_reads', 'dada2_merged_reads'),
        ('dada2_merged_reads', 'dada2_nonchim_reads'),
        ('dada2_nonchim_reads', 'target_reads'),
        ('p1_reads', 'p1_reads_pass'),
        ('p2_reads', 'p2_reads_pass'),
        ('raw_mosq_reads', 'mosq_reads'), # issue
        ('raw_mosq_targets_recovered', 'mosq_targets_recovered'), # issue
        ('raw_multiallelic_mosq_targets', 'multiallelic_mosq_targets')
        ]:
        assert (comb_df[colp] >= comb_df[coln]).all(), f'found less reads in {colp} than in {coln}'

    assert ((comb_df.overall_filter_rate >= 0) & (comb_df.overall_filter_rate <= 1)).all(), \
        'found overall_filter_rate outside of [0,1]'

    assert (comb_df.targets_recovered <= 64).all(), 'over 64 targets_recovered reported'

    for col in [
        'raw_mosq_targets_recovered', 'raw_multiallelic_mosq_targets',
        'multiallelic_mosq_targets', 'mosq_targets_recovered'
        ]:
        assert (comb_df[col] <= 62).all(), f'over 62 {col} reported'

    assert (comb_df.target_reads == comb_df.raw_mosq_reads + comb_df.p1_reads + comb_df.p2_reads).all(), \
        'target_reads does not match raw_mosq_reads + p1_reads + p2_reads'

    assert (comb_df.query('p1_reads_pass > 10')['p1_seqids_pass'].notna()).all(), \
        'not all P1 pass records supported by 10 reads have p1_seqids_pass recorded'
    assert ~((comb_df.p1_reads_pass < 10) & (comb_df.p1_reads_pass > 0)).any(), \
        'some P1 pass records supported by less than 10 reads'
    assert (comb_df.query('p1_reads_pass == 0')['p1_seqids_pass'].isna()).all(), \
        'record with zero p1_reads_pass has some p1_seqids_pass recorded'

    assert (comb_df.query('p2_reads_pass > 10')['p2_seqids_pass'].notna()).all(), \
        'not all P2 pass records supported by 10 reads have p2_seqids_pass recorded'
    assert ~((comb_df.p2_reads_pass < 10) & (comb_df.p2_reads_pass > 0)).any(), \
        'some P2 pass records supported by less than 10 reads'
    assert (comb_df.query('p2_reads_pass == 0')['p2_seqids_pass'].isna()).all(), \
        'record with zero p2_reads_pass has some p2_seqids_pass recorded'

def agg(args):

    setup_logging(verbose=args.verbose)

    logging.info('ANOSPP results merging data import started')
    
    run_id, comb_stats_df = prep_comb_stats(args.stats)
    # qc_df = pd.read_csv(args.qc, sep='\t')
    plasm_df = pd.read_csv(args.plasm, sep='\t')
    nn_df = pd.read_csv(args.nn, sep='\t')
    vae_df = pd.read_csv(args.vae, sep='\t')

    logging.info("merging results tables")

    # assert set(manifest_df.sample_id) == set(qc_df.sample_id), \
    #     'lanelets manifest and QC samples do not match'
    # comb_df = pd.merge(manifest_df, qc_df, how='inner')

    assert set(comb_stats_df.sample_id) == set(plasm_df.sample_id), \
        'plasm samples do not match comb stats'
    comb_df = pd.merge(comb_stats_df, plasm_df, how='inner')

    assert set(comb_df.sample_id) == set(nn_df.sample_id), \
        'NN samples do not match plasm, comb stats'
    comb_df = pd.merge(comb_df, nn_df, how='inner')

    assert vae_df.index.isin(comb_df.index).all(), \
        'VAE samples do not match NN, plasm, comb stats'
    comb_df = pd.merge(comb_df, vae_df, how='left')

    comb_df['nnovae_mosquito_species'] = comb_df.vae_species_call.fillna(comb_df.nn_species_call)
    is_nocall = comb_df['nnovae_mosquito_species'].isna()
    assert ~is_nocall.any(), \
        f'could not find none of NN or VAE call for {comb_df[is_nocall].index.to_list()}'
    
    # infer call method
    comb_df['nnovae_call_method'] = comb_df.nn_call_method
    comb_df.loc[
        comb_df.sample_id.isin(vae_df.sample_id),
        'nnovae_call_method'
    ] = 'VAE'

    if not args.force:
        validate_aggregation(comb_df)

    logging.info(f'writing merged results to {args.out}')
    comb_df.to_csv(args.out, sep='\t', index=False)


def main():
    
    parser = argparse.ArgumentParser('Merging ANOSPP run analysis results into a single file')
    parser.add_argument('-s', '--stats',
                        help='path to comb stats tsv generated by prep. Default: prep/comb_stats.tsv',
                        default='prep/comb_stats.tsv')
    parser.add_argument('-n', '--nn', 
                        help='path to NN assignment tsv. Default: nn/nn_assignment.tsv', 
                        default='nn/nn_assignment.tsv')
    parser.add_argument('-e', '--vae', 
                        help='path to VAE assignment tsv. Default: vae/vae_assignment.tsv',
                        default='vae/vae_assignment.tsv')
    parser.add_argument('-p', '--plasm', 
                        help='path to plasm assignment tsv. Default: plasm/plasm_assignment.tsv',
                        default='plasm/plasm_assignment.tsv')
    parser.add_argument('-o', '--out', 
                        help='Output aggregated sample metadata tsv. Default: anospp_results.tsv', 
                        default='anospp_results.tsv')
    parser.add_argument('-f', '--force', 
                        help='Skip aggregation validation', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true', default=True)
                        

    args = parser.parse_args()

    agg(args)


if __name__ == '__main__':
    main()