import argparse
import os
import re
import logging
import pandas as pd

from anospp_analysis.util import load_hap, load_comb_stats, setup_logging, hap_to_fa, PLASM_TARGETS
from anospp_analysis.vsearch import run_vsearch_sintax, parse_sintax

pd.set_option('future.no_silent_downcasting', True)

def estimate_contamination(
    hap_df,
    comb_stats_df,
    min_reads_source,
    min_samples_affected,
    min_cov_ratio,
):
    """
    Identify potential contamination from excessive haplotype sharing between
    high coverage sample (source) and many low coverage samples (affected).

    High confidence is given to contamination between samples sharing plate or well - 
    in ANOSPP, this also corresponds to sharing forward or reverse index 
    """

    ext_stats_df = comb_stats_df[[
        'sample_id',
        'plate_id',
        'well_id'
    ]]

    ext_hap_df = pd.merge(hap_df, ext_stats_df, on='sample_id', how='left')

    assert ~ext_hap_df['well_id'].isna().any(), 'failed to get well IDs'
    assert ~ext_hap_df['plate_id'].isna().any(), 'failed to get plate IDs'

    ext_hap_df['contamination_status'] = ''
    # ext_hap_df['contamination_confidence'] = ''

    for seqid, seqid_df in ext_hap_df.groupby('seqid'):
        # any potential source for seqid
        source_samples = (seqid_df.reads > min_reads_source)
        if source_samples.any():
            logging.info(f'contamination detected for {seqid}')
            # affected sample coverage cutoff
            max_reads_affected = seqid_df.reads.max() / min_cov_ratio
            affected_samples = (seqid_df.reads < max_reads_affected)
            if affected_samples.sum() >= min_samples_affected:
                # source and affected data
                src_df = seqid_df.loc[source_samples]
                tgt_df = seqid_df.loc[affected_samples]
                # sample & hap define positions in original df
                is_src_seqid = (ext_hap_df.sample_id.isin(src_df['sample_id']) & (ext_hap_df.seqid == seqid))
                is_tgt_seqid = (ext_hap_df.sample_id.isin(tgt_df['sample_id']) & (ext_hap_df.seqid == seqid))
                # set contamination statuses 
                # unclear - between max_reads_affected and min_reads_source
                ext_hap_df.loc[(ext_hap_df.seqid == seqid), 'contamination_status'] = 'unclear'
                ext_hap_df.loc[is_src_seqid, 'contamination_status'] = 'source'
                # confidence - low without plate/well match
                ext_hap_df.loc[is_tgt_seqid, 'contamination_status'] = 'affected_possible'
                for _, src_row in src_df.iterrows():
                    # affected samples sharing plate or well with source
                    same_plate_tgt_samples = tgt_df.loc[tgt_df.plate_id == src_row.plate_id, 'sample_id']
                    if same_plate_tgt_samples.shape[0] > 0:
                        hc_tgt_haps = (ext_hap_df.sample_id.isin(same_plate_tgt_samples) & (ext_hap_df.seqid == seqid))
                        ext_hap_df.loc[hc_tgt_haps, 'contamination_status'] = 'affected_plate_sharing'
                    same_well_tgt_samples = tgt_df.loc[tgt_df.well_id == src_row.well_id, 'sample_id']
                    if same_well_tgt_samples.shape[0] > 0:
                        hc_tgt_haps = (ext_hap_df.sample_id.isin(same_well_tgt_samples) & (ext_hap_df.seqid == seqid))
                        ext_hap_df.loc[hc_tgt_haps, 'contamination_status'] = 'affected_well_sharing'
    

    out_hap_df = ext_hap_df.drop(columns=['plate_id','well_id'])

    return out_hap_df

def summarise_samples(sum_hap_df, comb_stats_df, targets=PLASM_TARGETS):

    # base sample table
    sum_samples_df = (
        comb_stats_df[
            [
                'sample_id',
                'sample_name',
                'lims_plate_id',
                'lims_well_id',
                'plate_id',
                'well_id',
            ]
        ]
        .set_index('sample_id')
        .copy()
    )

    for t in targets:

        t_df = (
            sum_hap_df[sum_hap_df.target == t]
            .sort_values('reads', ascending=False)
        )

        # fill hap catagories - from lowest to highest priority
        t_df['category'] = 'pass'
        t_df.loc[t_df['locov'], 'category'] = 'locov'
        t_df.loc[
            (t_df['contamination_status'] == 'affected_possible'),
            'category'
            ] = 'contam_possible'
        t_df.loc[
            (t_df['contamination_status'] == 'affected_well_sharing'),
            'category'
            ] = 'contam_well'
        t_df.loc[
            (t_df['contamination_status'] == 'affected_plate_sharing'),
            'category'
            ] = 'contam_plate'

        for col, outcol in [
                ('seqid', f'{t}_seqids'),
                ('reads', f'{t}_seqids_reads'),
                ('category', f'{t}_seqids_categories'),
                # ('genus', f'{t}_seqids_genus'),
                # ('genus_bootstrap', f'{t}_seqids_genus_bootstrap'),
                ('group', f'{t}_seqids_group'),
                ('group_bootstrap', f'{t}_seqids_group_bootstrap'),
                ('species', f'{t}_seqids_species'),
                ('species_bootstrap', f'{t}_seqids_species_bootstrap'),
            ]:
                agg_df = t_df[['sample_id',col]].copy()
                if pd.api.types.is_numeric_dtype(agg_df[col]):
                    agg_df[col] = agg_df[col].fillna(0).astype(str)
                else:
                    agg_df[col] = agg_df[col].fillna('')
                sum_samples_df[outcol] = (
                    agg_df.groupby('sample_id')[col]
                    .agg(';'.join)
                    .reindex(sum_samples_df.index)
                    .fillna('')
                )

    detection = {
        'plasm_taxa':{},
        'plasm_labels':{},
        'plasm_detection':{},
        'plasm_groups_detected':{},
        'plasm_species_detected':{}
    }
    for sample_id, sample_hap_df in sum_hap_df.groupby('sample_id'):

        # higher coverage taxa to be reported first
        sample_hap_df = sample_hap_df.sort_values(by='reads', ascending=False)

        sample_spp = sample_hap_df['species'].unique()
        sample_grp = sample_hap_df['group'].unique()

        if len(sample_spp) > len(sample_grp):
            logging.warning(f'multiple species of same group detected for sample {sample_id}: analysis on group level')
            tax_col = 'group'
        else:
            tax_col = 'species'
        
        for k in detection.keys():
            detection[k][sample_id] = []
        # detection status recorded once for sample - default at not detected
        detection['plasm_detection'][sample_id] = ['not_detected']

        for tax, tax_sample_hap_df in sample_hap_df.groupby(tax_col):
            tax_labels = []
            # mixed species
            if tax_col == 'group':
                tax_labels += ['mixed-spp']
            tax_targets = tax_sample_hap_df['target'].unique()
            # single target
            if len(tax_targets) == 1:
                tax_labels += [f'{tax_targets[0]}-only']
            # masks for other labels - not from categories as these can overlap
            is_contam = tax_sample_hap_df['contamination_status'].isin((
                'affected_well_sharing',
                'affected_plate_sharing'
            ))
            # at least low confidence contamination
            possible_contam = tax_sample_hap_df['contamination_status'].str.startswith('affected')
            is_locov = tax_sample_hap_df['locov']
            is_multiallelic = (tax_sample_hap_df['target'].value_counts() > 1)
            is_hicov = (tax_sample_hap_df['contamination_status'] == 'source')
            # all haplotypes are contamination - any unique haplotypes indicative of real infection
            if is_contam.all():
                tax_labels += ['contam']
            # all haplotypes at least low confidence contamination
            elif possible_contam.all():
                tax_labels += ['contam-possible']
            # all haplotypes low coverage - any higher coverage haplotypes indicative of real infection
            if is_locov.all():
                tax_labels += ['locov']
            # any multiallelics - mixed infection or high coverage
            if is_multiallelic.any():
                tax_labels += ['multiallelic']
            # any high coverage haplotypes 
            if is_hicov.any():
                tax_labels += ['hicov']
            # unmarked with any issues above is pass
            if len(tax_labels) == 0:
                tax_labels += ['pass']

            # aggregate labels into status - from high to low priority
            tax_status = 'undefined'
            for tgt in PLASM_TARGETS:
                if f'{tgt}-only' in tax_labels:
                    tax_status = 'not_detected'
                    break
            else:
                if 'contam' in tax_labels:
                    tax_status = 'not_detected'
                elif 'contam-possible' in tax_labels:
                    tax_status = 'not_detected'
                elif 'locov' in tax_labels:
                    tax_status = 'not_detected'
                elif 'mixed-spp' in tax_labels:
                    tax_status = 'detected'
                elif 'multiallelic' in tax_labels:
                    tax_status = 'detected'
                elif 'hicov' in tax_labels:
                    tax_status = 'detected'
                elif 'pass' in tax_labels:
                    tax_status = 'detected'
                else:
                    raise ValueError(f'failed to parse status for {sample_id}, {tax}: {tax_labels}')
            # record statuses
            detection['plasm_taxa'][sample_id].append(tax)
            detection['plasm_labels'][sample_id].append('_'.join(tax_labels))
            if tax_status != 'not_detected':
                # update detection status
                detection['plasm_detection'][sample_id] = ['detected']
                # only one group possible
                detection['plasm_groups_detected'][sample_id].append(tax_sample_hap_df['group'].iloc[0])
                # multiple species per group possible - record top botstrap
                top_sp = tax_sample_hap_df.sort_values('species_bootstrap', ascending=False)['species'].iloc[0]
                detection['plasm_species_detected'][sample_id].append(top_sp)
        
        for k in detection.keys():
            detection[k][sample_id] = ';'.join(detection[k][sample_id])

    for annotation in detection.keys():
        fill_value = ('not_detected' if annotation == 'plasm_detection' else '')
        sum_samples_df[annotation] = pd.Series(detection[annotation])
        sum_samples_df[annotation] = sum_samples_df[annotation].fillna(fill_value)

    return sum_samples_df

def plot_plate_view(plasm_df, out_fn, reference_colours, lims_plate=True, title=None):

    from bokeh.plotting import figure, output_file, save
    from bokeh.models import ColumnDataSource, Span
    from bokeh.transform import factor_cmap
    from bokeh.models.tools import HoverTool
    from bokeh.transform import dodge

    '''
    Plots interactive plate map with plasm annotations
    '''
    # editable copy of sample info
    plasm_df = plasm_df.copy()

    # set the output filename
    output_file(out_fn)

    #extract the column and generate the row values
    if lims_plate:
        cols = list('ABCDEFGHIJKLMNOP')
        rows = [str(x) for x in range(1, 25)]
        plasm_df['col'] = plasm_df['lims_well_id'].str[0]
        plasm_df['row'] = plasm_df['lims_well_id'].str[1:]
    else:
        cols = list('ABCDEFGH')
        rows = [str(x) for x in range(1, 13)]
        plasm_df['col'] = plasm_df['well_id'].str[0]
        plasm_df['row'] = plasm_df['well_id'].str[1:]

    # display values
    plasm_df['P1_seqids_disp'] = plasm_df['P1_seqids'].str.replace(';.*', '...', regex=True)
    plasm_df['P2_seqids_disp'] = plasm_df['P2_seqids'].str.replace(';.*', '...', regex=True)
    plasm_df['comb_seqids_disp'] = plasm_df['P1_seqids_disp'] + '\n' + plasm_df['P2_seqids_disp']

    # colour by taxon and detection status
    plasm_df['colour_group'] = 'not_detected'
    plasm_df.loc[plasm_df['plasm_groups_detected'].str.contains(';'), 'colour_group'] = 'mixed_infection'
    plasm_df.loc[plasm_df['plasm_taxa'] != '', 'colour_group'] = 'fail_detection'
    plasm_df.loc[plasm_df['plasm_groups_detected'] != '', 'colour_group'] = plasm_df['plasm_groups_detected']

    # colour for inferred statuses
    reference_colours['not_detected'] = '#ffffff'
    reference_colours['mixed_infection'] = '#cfcfcf'
    reference_colours['fail_detection'] = '#efefef'

    #load the dataframe into bokeh
    source = ColumnDataSource(plasm_df)

    #set up the figure
    p = figure(
        width=(1300 if lims_plate else 900),
        height=(600 if lims_plate else 400),
        title=title,
        x_range=rows,
        y_range=list(reversed(cols)),
        toolbar_location=None,
        tools=[HoverTool(), 'pan', 'wheel_zoom', 'reset']
        )

    # add grid lines
    for v in range(len(rows)):
        line_width = 2 if (v % 2 == 0 and lims_plate) else 1
        vline = Span(location=v, dimension='height', line_color='black', line_width=line_width)
        p.renderers.extend([vline])

    for h in range(len(cols)):
        line_width = 2 if (h % 2 == 0 and lims_plate) else 1
        hline = Span(location=h, dimension='width', line_color='black', line_width=line_width)
        p.renderers.extend([hline])
    
    #add the rectangles
    p.rect(
        'row',
        'col',
        0.95,
        0.95,
        source=source,
        fill_alpha=.9,
        legend_field='colour_group',
        color=factor_cmap('colour_group', palette=list(reference_colours.values()), factors=list(reference_colours.keys()))
        )

    #add the species count text for each field
    text_props = {'source': source, 'text_align': 'left', 'text_baseline': 'middle'}
    x = dodge('row', -0.4, range=p.x_range)
    r = p.text(x=x, y='col', text='comb_seqids_disp', **text_props)
    r.glyph.text_font_size = '10px'
    r.glyph.text_font_style = 'bold'

    #set up the hover value
    p.add_tools(HoverTool(tooltips=[
        ('sample id', '@{sample_id}'),
        ('Parasite detection', '@plasm_detection'),
        ('Parasite groups detected', '@plasm_groups_detected'),
        ('Parasite species detected', '@plasm_species_detected'),
        ('Parasite taxa', '@plasm_taxa'),
        ('Parasite labels', '@plasm_labels'),
        ('P1 haplotype IDs', '@P1_seqids'),
        ('P1 reads per haplotype', '@P1_seqids_reads'),
        ('P1 categories per haplotype', '@P1_seqids_categories'),
        ('P1 species per haplotype', '@P1_seqids_species'),
        ('P1 species bootstrap per haplotype', '@P1_seqids_species_bootstrap'),
        ('P2 haplotype IDs', '@P2_seqids'),
        ('P2 reads per haplotype', '@P2_seqids_reads'),
        ('P2 categories per haplotype', '@P2_seqids_categories'),
        ('P2 species per haplotype', '@P2_seqids_species'),
        ('P2 species bootstrap per haplotype', '@P2_seqids_species_bootstrap'),
        
    ]))

    #set up the rest of the figure and save the plot
    p.outline_line_color = 'black'
    p.grid.grid_line_color = None
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = 'vertical'
    p.legend.click_policy='hide'
    p.add_layout(p.legend[0], 'right') 
    save(p)

def empty_haps():

    cols = [
        # from haps
        'sample_id','target','consensus','reads','total_reads','reads_fraction','nalleles','seqid',
        # from pre-processing
        'locov',
        # from sintax
        'strand',
        'domain','domain_bootstrap','phylum','phylum_bootstrap','order','order_bootstrap','family','family_bootstrap',
        'genus','genus_bootstrap','subgenus','subgenus_bootstrap','group','group_bootstrap','species','species_bootstrap',
        ]

    return pd.DataFrame(columns=cols)

def plasm(args):

    # Set up logging and create output directories
    setup_logging(verbose=args.verbose)

    # reference checks
    logging.info('Checking plasm reference data')
    reference_path = os.path.abspath(args.reference_path.rstrip('/'))
    reference_version = reference_path.split('/')[-1]
    assert os.path.isdir(reference_path), f'reference directory does not exist at {reference_path}'
    assert re.match(r'^plasmv\d', reference_version), f'{reference_version} not recognised as plasm ref version'
    assert reference_version != 'plasmv1', 'plasmv1 not supported since anospp-analysis v0.5.0'
    sintax_dbs = {}
    for tgt in PLASM_TARGETS:
        sintax_db_path = f'{reference_path}/sintax_db_{tgt}.fasta'
        assert os.path.isfile(sintax_db_path), f'SINTAX database does not exist at {sintax_db_path}'
        sintax_dbs[tgt] = sintax_db_path
    sintax_ranks = pd.read_csv(f'{reference_path}/sintax_ranks.tsv', sep='\t', index_col=0)['plasm_rank'].to_dict()
    reference_colours = pd.read_csv(f'{reference_path}/plasm_colours.tsv', sep='\t', index_col=0)['colour'].to_dict()

    os.makedirs(args.outdir, exist_ok=True)

    logging.info('ANOSPP plasm data import started')
    hap_df = load_hap(args.haplotypes)
    run_id, comb_stats_df = load_comb_stats(args.stats)

    plasm_hap_df = hap_df[hap_df['target'].isin(PLASM_TARGETS)].copy()

    sintax_dfs = []
    for tgt in PLASM_TARGETS:
        tgt_plasm_hap_df = plasm_hap_df.query('target == @tgt')
        uniq_tgt_plasm_hap_df = tgt_plasm_hap_df[['seqid','consensus']].drop_duplicates()
        if uniq_tgt_plasm_hap_df.shape[0] == 0:
            logging.info(f'no {tgt} haplotypes, skipping SINTAX')
            sintax_dfs.append(pd.DataFrame())
            continue
        uniq_tgt_plasm_hap_fa = f'{args.outdir}/raw_{tgt}_haps.fa'
        logging.info(f'writing unique {tgt} haplotypes to {uniq_tgt_plasm_hap_fa}')
        hap_to_fa(uniq_tgt_plasm_hap_df, uniq_tgt_plasm_hap_fa)

        tgt_sintax_tsv = f'{args.outdir}/sintax_out_{tgt}.tsv'
        tgt_sintax_log = f'{args.outdir}/sintax_out_{tgt}.log'
        logging.info(f'running SINTAX assignment for {uniq_tgt_plasm_hap_fa} against {sintax_dbs[tgt]}, writing to {tgt_sintax_tsv}')
        o, e = run_vsearch_sintax(
            uniq_tgt_plasm_hap_fa,
            sintax_dbs[tgt],
            tgt_sintax_tsv,
            threads=1
        )
        with open(tgt_sintax_log, 'w') as log:
            log.write(e)
        logging.info(f'parsing SINTAX output {tgt_sintax_tsv}')
        sintax_df = parse_sintax(tgt_sintax_tsv, sintax_ranks)
        sintax_df['target'] = tgt
        sintax_dfs.append(sintax_df)
    
    sintax_df = pd.concat(sintax_dfs)

    annotated_hap_fn = f'{args.outdir}/plasm_hap_summary.tsv'
    # annotated_sample_fn = f'{args.outdir}/plasm_sample_summary.tsv'
    plasm_assignment_fn = f'{args.outdir}/plasm_assignment.tsv' 
    if sintax_df.shape[0] > 0:
        annotated_hap_df = pd.merge(plasm_hap_df, sintax_df, on=['seqid','target'])
        logging.info(f'writing annotated haplotypes to {annotated_hap_fn}')
    else:
        logging.warning('no Plasmodium haplotypes to annotate, creating empty haplotypes file')
        annotated_hap_df = empty_haps()

    # mark locov 
    annotated_hap_df['locov'] = False
    for tgt, mincov in zip(PLASM_TARGETS, (args.filter_p1, args.filter_p2)):
        annotated_hap_df.loc[
            (annotated_hap_df['target'] == tgt) & (annotated_hap_df['reads'] < mincov),
            'locov'
        ] = True

    if args.estimate_contamination:
        logging.info('estimating cross-contamination')
        annotated_hap_df = estimate_contamination(
            annotated_hap_df,
            comb_stats_df,
            min_reads_source=args.contam_min_reads_source,
            min_samples_affected=args.contam_min_samples_affected,
            min_cov_ratio=args.contam_min_cov_ratio,
        )
    else:
        logging.info('skipping cross-contamination analysis')
        annotated_hap_df['contamination_status'] = ''
        annotated_hap_df['contamination_confidence'] = ''

    annotated_hap_df.to_csv(annotated_hap_fn, sep='\t', index=False)

    logging.info('summarising sample level plasm annotations')
    annotated_sample_df = summarise_samples(
        annotated_hap_df, 
        comb_stats_df,
        targets=PLASM_TARGETS,
    )
    
    if args.interactive_plotting:

        for plate in annotated_sample_df.plate_id.unique():

            logging.info(f'plotting interactive plate view for {plate}')
            
            out_fn = f'{args.outdir}/plasm_plate_{plate}.html'
            title = f'Plasmodium species composition run {run_id}, plate {plate}'
            plot_df = annotated_sample_df[annotated_sample_df.plate_id == plate]
            plot_plate_view(plot_df, out_fn, reference_colours, lims_plate=False, title=title)
        for lims_plate in annotated_sample_df.lims_plate_id.unique():
            logging.info(f'plotting interactive plate view for LIMS plate {lims_plate}')
            
            out_fn = f'{args.outdir}/plasm_lims_plate_{lims_plate}.html'
            title = f'Plasmodium species composition run {run_id}, LIMS plate {lims_plate}'
            plot_df = annotated_sample_df[annotated_sample_df.lims_plate_id == lims_plate]
            plot_plate_view(plot_df, out_fn, reference_colours, lims_plate=True, title=title)

    annotated_sample_df['plasm_ref'] = reference_version
    annotated_sample_df.reset_index(inplace=True)
    annotated_sample_df.columns = annotated_sample_df.columns.str.lower()
    # logging.info(f'writing all sample level info to {annotated_sample_fn}')
    # annotated_sample_df.to_csv(annotated_sample_fn, sep='\t', index=False)
    logging.info(f'writing sample level plasm assignment to {plasm_assignment_fn}')
    # explicitly listing all cols rather than dropping plate/well/sample info used for plotting
    annotated_sample_df[[
        'sample_id',
        'p1_seqids',
        'p1_seqids_reads',
        'p1_seqids_categories',
        # 'p1_seqids_genus',
        # 'p1_seqids_genus_bootstrap',
        'p1_seqids_group',
        'p1_seqids_group_bootstrap',
        'p1_seqids_species',
        'p1_seqids_species_bootstrap',
        'p2_seqids',
        'p2_seqids_reads',
        'p2_seqids_categories',
        # 'p2_seqids_genus',
        # 'p2_seqids_genus_bootstrap',
        'p2_seqids_group',
        'p2_seqids_group_bootstrap',
        'p2_seqids_species',
        'p2_seqids_species_bootstrap',
        'plasm_taxa',
        'plasm_labels',
        'plasm_detection',
        'plasm_groups_detected',
        'plasm_species_detected',
        'plasm_ref'
    ]].to_csv(plasm_assignment_fn, sep='\t', index=False)

    annotated_sample_df.to_csv(plasm_assignment_fn, sep='\t', index=False)

    logging.info('ANOSPP plasm complete')


def main():

    parser = argparse.ArgumentParser("Plasmodium species assignment for ANOSPP data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file generated by prep', required=True)
    parser.add_argument('-s', '--stats', help='Comb stats tsv file generated by prep', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: plasmv2', default='plasmv2')
    parser.add_argument('-r', '--reference_path', 
                        help='Path to plasm reference directory containing SiNTAX reference fasta for each amplicon and colour scheme', 
                        required=True)
    parser.add_argument('-f', '--filter_p1',
                        help='Minimum read support for P1 haplotypes to be included in sample summary. Default: 10',
                        default=10, type=int)
    parser.add_argument('-g', '--filter_p2', 
                        help='Minimum read support for P2 haplotypes to be included in sample summary. Default: 10', 
                        default=10, type=int)
    parser.add_argument('-e','--estimate_contamination', 
                        help='Perform plate/well based contamination removal optimised for ANOSPP Illumina workflow. Disabled by default', 
                        action='store_true')
    parser.add_argument('--contam_min_reads_source', 
                        help='Minimum number of reads in a source sample for same plate/well contamination. Default: 10000',  
                        default=10000, type=int)
    parser.add_argument('--contam_min_samples_affected', 
                        help='Minimum number of samples affected by plate/well contamination to be considered. Default: 4',  
                        default=4, type=int)
    parser.add_argument('--contam_min_cov_ratio', 
                        help='Maximum coverage ratio between source and target of contamination. Default: 100',  
                        default=100, type=float)
    parser.add_argument('-i', '--interactive_plotting', 
                        help='Create interactive plots of species composition across plates', 
                        action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')


    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    plasm(args)


if __name__ == '__main__':
    main()

