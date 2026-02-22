import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import OrderedDict
from scipy.optimize import curve_fit
import math
import os
import argparse
import logging
import pandas as pd

from anospp_analysis.util import setup_logging, load_comb_stats, load_hap, well_ordering, human_format, MOSQ_TARGETS, CUTADAPT_TARGETS

def plot_target_balance(hap_df, run_id):

    logging.info('plotting targets balance')

    reads_per_sample_target = hap_df \
        .groupby(['sample_id', 'target'], observed=False)['reads'].sum().reset_index()
    
    figsize = (hap_df['target'].nunique() * 0.3, 6)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    fig.suptitle(f'Amplicon coverage distribution per sample for run {run_id}')
    sns.stripplot(
        data=reads_per_sample_target,
        x = 'target',
        y = 'reads',
        hue = 'target',
        alpha = .1,
        jitter = .3,
        ax = ax
        )
    # ax.get_legend().remove()
    ax.set_yscale('log')
    # ax.set_ylim(bottom=0.5)
    ax.set_ylabel('reads')
    ax.set_xlabel('target')
    ax.axhline(10, c='silver', alpha=.5)
    ax.tick_params(axis='x', rotation=90)

    return fig, ax

def plot_allele_balance(hap_df, run_id, anospp=True):
    
    logging.info('plotting allele balance and coverage')

    if anospp:
        hap_df = hap_df[hap_df.target.isin(MOSQ_TARGETS)]

    is_het = (hap_df.reads_fraction < 1)
    het_frac = hap_df[is_het].reads_fraction
    het_reads_log = hap_df[is_het].reads.apply(lambda x: np.log10(x))
    het_plot = sns.jointplot(
        x=het_frac,
        y=het_reads_log,
        kind="hist",
        height=8
        )
    het_plot.ax_joint.set_ylabel('reads (log10)')
    het_plot.ax_joint.set_xlabel('allele fraction')
    het_plot.ax_joint.axhline(1, c='silver', alpha=.5)
    het_plot.ax_joint.axvline(0.1, c='silver', alpha=.5)
    # funky title setting
    extra = ' in mosquito targets' if anospp else ''
    het_plot.fig.suptitle(f'Allele fractions and coverage{extra} for run {run_id}')
    het_plot.fig.tight_layout()
    het_plot.fig.subplots_adjust(top=0.95) # Reduce plot to make room
    
    return het_plot

def plot_well_balance(comb_stats_df, run_id):

    logging.info('plotting wells balance')

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f'Reads per well for run {run_id}')
    sns.boxplot(
            data=comb_stats_df,
            x = 'well_id',
            y = 'total_reads',
            ax = ax
            )
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=90)

    return fig, ax

def plot_sample_filtering(comb_stats_df, run_id, anospp=True):
    
    logging.info('plotting per-sample filtering barplots')

    # comb_stats_df colname : legend label
    dada2_cols = OrderedDict([
        ('total_reads', 'removed as readthrough'),
        ('dada2_input_reads', 'removed by filterAndTrim'), 
        ('dada2_filtered_reads', 'removed by denoising'),
        ('dada2_denoised_reads', 'removed by merging'), 
        ('dada2_merged_reads', 'removde by rmchimera'), 
        ('dada2_nonchim_reads', 'unassigned to amplicons'),
        # legacy post-filter disabled in prod
        # ('dada2_final_reads', 'unassigned to amplicons'),
        ])

    if anospp:
        dada2_cols['target_reads'] = 'Plasmodium reads'
        dada2_cols['raw_mosq_reads'] = 'mosquito reads'
    else:
        dada2_cols['target_reads'] = 'target reads'
    
    plates = comb_stats_df.plate_id.unique()
    nplates = len(plates)
    fig, axs = plt.subplots(nplates, 1, figsize=(20, 4 * nplates), constrained_layout=True)
    fig.suptitle(f'Read filtering stats for run {run_id}', fontsize=20)
    if nplates == 1:
        axs = [axs]
    for plate, ax in zip(plates, axs):
        plot_df = comb_stats_df[comb_stats_df.plate_id == plate].copy()
        plot_df['well_id'] = well_ordering(plot_df['well_id'])
        plot_df.sort_values(by='well_id', inplace=True)
        for i, col in enumerate(dada2_cols.keys()):
            sns.barplot(
                x='sample_id',
                y=col,
                data=plot_df,
                color=sns.color_palette()[i],
                ax=ax,
                label=dada2_cols[col],
                width=1
                )
        ax.set_xticks(range(plot_df.shape[0]))
        ax.set_xticklabels(plot_df['sample_name'])
        ax.set_xlabel('sample_name')
        ax.tick_params(axis='x', rotation=90)
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5, top=max(comb_stats_df['total_reads']))
        ax.set_ylabel('reads')
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=14)
        ax.set_title(plate)

    return fig, axs

def plot_plate_stats(comb_stats_df, run_id, lims_plate=False):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'
    size = '384-well' if lims_plate else '96-well'

    logging.info('plotting plate stats')
    
    fig, axs = plt.subplots(
        3, 1,
        figsize=(10,15),
        constrained_layout=True
        )
    fig.suptitle(f'Key {size} plate performance stats for run {run_id}')

    sns.stripplot(
        data=comb_stats_df,
        y='target_reads',
        x=plate_col,
        hue=plate_col,
        alpha=.3,
        jitter=.35,
        ax=axs[0]
        )
    axs[0].set_yscale('log')
    axs[0].set_ylim(bottom=1)
    axs[0].set_xlabel('')
    axs[0].set_xticklabels([])
    # 1000 reads cutoff
    axs[0].axhline(1000, c='silver', alpha=.5)
    
    sns.stripplot(
        data=comb_stats_df,
        y='raw_mosq_targets_recovered',
        x=plate_col,
        hue=plate_col,
        alpha=.3,
        jitter=.35,
        ax=axs[1]
        )
    axs[1].set_ylim(bottom=0, top=62)
    axs[1].set_xticklabels([])
    axs[1].set_xlabel('')
    # 10/50 targets cutoff - nn/vae
    axs[1].axhline(10, c='silver', alpha=.5)
    axs[1].axhline(50, c='silver', alpha=.5)

    sns.stripplot(
        data=comb_stats_df,
        y='overall_filter_rate',
        x=plate_col,
        hue=plate_col,
        alpha=.3,
        jitter=.35,
        ax=axs[2]
        )
    axs[2].set_ylim(bottom=0, top=1)
    # 50% filtering cutoff
    axs[2].axhline(.5, c='silver', alpha=.5)
    # for ax in axs:
    #     ax.get_legend().remove()
    plt.xticks(rotation=90)

    return fig, axs

def plot_plate_summaries(comb_stats_df, run_id, lims_plate=False):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'
    size = '384-well' if lims_plate else '96-well'

    logging.info(f'plotting success summaries by {plate_col}')
    
    # success rate definition
    comb_stats_df['over 1000 mosquito reads'] = comb_stats_df.raw_mosq_reads > 1000
    comb_stats_df['over 10 targets'] = comb_stats_df.targets_recovered > 10
    comb_stats_df['over 50% reads retained'] = comb_stats_df.overall_filter_rate > .5

    plates = comb_stats_df[plate_col].unique()
    nplates = comb_stats_df[plate_col].nunique()

    sum_df = comb_stats_df.groupby(plate_col) \
        [['over 1000 mosquito reads', 'over 10 targets', 'over 50% reads retained']].sum()
    y = comb_stats_df.groupby(plate_col)['over 1000 mosquito reads'].count()
    sum_df = sum_df.divide(y, axis=0).reindex(plates)

    fig, ax = plt.subplots(
        1, 1, 
        figsize=(nplates * .5 + 2.5, 4),
        constrained_layout=True
        )
    fig.suptitle(f'{size} plate success rate for run {run_id}', fontsize=10)
    sns.heatmap(sum_df.T, annot=True, ax=ax, vmax=1, vmin=0)
    plt.xticks(rotation=90)

    return fig, ax

def plot_sample_success(comb_stats_df, run_id, anospp=True):

    logging.info('plotting sample success')

    ycol = 'raw_mosq_reads' if anospp else 'target_reads'
    tgtcol = 'raw_mosq_targets_recovered' if anospp else 'targets_recovered'

    fig, axs = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
    fig.suptitle(f'Key stats covariation for run {run_id}')
    for xcol, ax in zip((tgtcol, 'overall_filter_rate'), axs):
        sns.scatterplot(
            data=comb_stats_df,
            x=xcol,
            y=ycol,
            hue='plate_id',
            alpha=.5, 
            ax=ax
            )
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
        ax.axhline(1000, c='silver', alpha=.5)
    
    axs[0].set_xlim(left=0)
    if anospp:
        axs[0].axvline(10, c='silver', alpha=.5)
        axs[0].axvline(50, c='silver', alpha=.5)
        axs[0].set_xlim(right=62)
    axs[1].axvline(.5, c='silver', alpha=.5)
    axs[1].set_xlim(left=0, right=1)
    axs[1].get_legend().remove()

    return fig, axs

def plot_plasm_balance(comb_stats_df, run_id):

    logging.info('plotting Plasmodium read balance')

    max_plasm_reads = max(
        max(
            comb_stats_df.p1_reads.max(), 
            comb_stats_df.p2_reads.max()
            ),
        1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.suptitle(f'P1/P2 coverage balance for run {run_id}')
    sns.scatterplot(
        # display P1 or P2-only samples
        data=comb_stats_df.replace(0, 0.5),
        x='p1_reads',
        y='p2_reads',
        hue='plate_id',
        alpha=.5, 
        ax=ax
        )
    # hard filter cutoff
    ax.axhline(10, c='silver', alpha=.5)
    ax.axvline(10, c='silver', alpha=.5)
    # contamination affected sample cutoff
    ax.axhline(100, c='green', alpha=.5, linestyle='dashed')
    ax.axvline(100, c='green', alpha=.5, linestyle='dashed')
    # contamination source sample cutoff
    ax.axhline(10000, c='red', alpha=.5, linestyle='dashed')
    ax.axvline(10000, c='red', alpha=.5, linestyle='dashed')
    ax.plot(
        [0.4, max_plasm_reads],
        [0.4, max_plasm_reads],
        color='silver',
        linestyle='dashed',
        alpha=.5
        )
    ax.set_yscale('log')
    ax.set_xscale('log')

    return fig, ax

def plot_plate_heatmap(comb_stats_df, col, run_id, lims_plate=False, **heatmap_kwargs):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'
    well_col = 'lims_well_id' if lims_plate else 'well_id'
    size = '384-well' if lims_plate else '96-well'
    plate_width = 14 if lims_plate else 7
    plate_height = 8 if lims_plate else 5
    ncols = 1 if lims_plate else 2

    logging.info(f'plotting heatmap for {col} by {plate_col}')

    plates = comb_stats_df[plate_col].unique()
    nplates = comb_stats_df[plate_col].nunique()
    # ceil
    nrows = -(nplates // -ncols)

    comb_stats_df['row'] = comb_stats_df[well_col].str.slice(0, 1)
    comb_stats_df['col'] = comb_stats_df[well_col].str.slice(1).astype(int)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(plate_width * ncols, plate_height * nrows), 
        constrained_layout=True
        )
    fig.suptitle(f'Heatmap of {col} for {size} plates in run {run_id}', fontsize=16)
    if nplates == 1:
        axs = np.array([axs])
    for plate, ax in zip(plates, axs.flatten()):
        pdf = comb_stats_df[comb_stats_df[plate_col] == plate]
        hdf = pdf.pivot(index='row', columns='col', values=col)
        # read counts adjustments
        if 'fmt' in heatmap_kwargs.keys():
            if heatmap_kwargs['fmt'] == '':
                # human formatted labels
                heatmap_kwargs['annot'] = hdf.map(human_format)
                # handling of zero counts
                hdf = hdf.replace(0, 0.1)
        sns.heatmap(hdf, ax=ax, **heatmap_kwargs)
        if lims_plate:
            ax.hlines([i * 2 for i in range(9)], 0, 24, colors='k')
            ax.vlines([j * 2 for j in range(13)], 0, 16, colors='k')
        title = f'{plate} {col}'
        ax.set_title(title)

    return fig, axs

def plot_het_cov(hap_df, title='Total', run_id=None):

    logging.info('plotting heterozygosity vs coverage')

    nalleles_df = hap_df.pivot_table(
        index='sample_id',
        columns='target',
        values='nalleles',
        aggfunc='max',
        fill_value=0,
        observed=False)
    het_df = pd.DataFrame({
        'reads':hap_df.groupby(['sample_id'])['reads'].sum(),
        'targets':hap_df.groupby(['sample_id'])['target'].nunique(),
        'het_targets':(nalleles_df > 1).sum(axis=1)
    })
    het_df['het_rate'] = het_df['het_targets'] / het_df['targets']

    def logistic(x, L, k, x0):
        """
        Logistic function that can plateau:
        L = maximum value (plateau)
        k = growth rate
        x0 = the x-value of the sigmoid's midpoint
        """
        return L / (1 + np.exp(-k * (x - x0)))
    # fit the logistic curve to the data using curve_fit
    popt, _ = curve_fit(logistic, het_df.reads.fillna(-1), het_df.het_rate, p0=[.4, 0.0004, 3000])
    # popt contains the optimal parameters L, k, x0
    L, k, x0 = popt
    # set graph limit to nearest 1000
    xmax=math.ceil(float(het_df.reads.max()) / 1000) * 1000
    ymax=np.max(het_df.het_rate)
    x = np.arange(xmax) * 100
    y = logistic(x, L, k, x0)
    # cutoff values
    cutoffs = {}
    # for r in (.90,.95,.99):
    # find the index of 0.9  of limit of curve-fitted line and use the same index to find x value
    y_index=len(list(filter(lambda x: float(x) < float(L) * 0.9, list(y))))
    # transform into read counts
    x_threshold = x[y_index]

    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    ax.scatter(het_df.reads, het_df.het_rate, alpha=.1)
    ax.plot(x, y, c='r', label=f'Logistic het, max {L:.3f}')
    ax.vlines(x_threshold, 0, 0.95 * ymax, color='k', ls=':', label=f'Raw reads at 90% het: {x_threshold}')
    ax.set_xlim(-500,xmax)
    # ax.set_xscale('log')
    ax.legend()
    ax.set_ylabel('heterozygosity rate')
    ax.set_xlabel('reads')
    title = f'{title} reads and heterozygosity'
    if run_id is not None:
        title += f' for run {run_id}'
    ax.set_title(title)

    return fig, ax, L, x_threshold

def plot_cov(comb_stats_df, run_id):

    logging.info('plotting coverage')

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for col, ax in zip(['total_reads', 'raw_mosq_reads'], axs):

        data = comb_stats_df[col]
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        quantiles = data.quantile([0.25, 0.75, 0.95])
        q25, q75, q95 = quantiles[0.25], quantiles[0.75], quantiles[0.95]
        
        # Alt plotting options
        # ax.set_xscale('log')
        # sns.kdeplot(x=data.replace(0,0.5), ax=ax)
        # sns.boxenplot(x=data.replace(0,0.5), ax=ax, alpha=.1)
        # sns.stripplot(x=data.replace(0,0.5), jitter=False, ax=ax)

        # Plot histogram
        ax.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add horizontal lines for mean, median, and quantiles
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.0f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.0f}')
        ax.axvline(q25, color='purple', linestyle='-.', linewidth=1.5, label=f'25% Quantile: {q25:.0f}')
        ax.axvline(q75, color='orange', linestyle='-.', linewidth=1.5, label=f'75% Quantile: {q75:.0f}')
        ax.axvline(q95, color='blue', linestyle='-.', linewidth=1.5, label=f'95% Quantile: {q95:.0f}')
        
        # Add labels and legend
        ax.set_title(f'Run {run_id} {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()

    return fig, axs

def qc(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok=True)
    
    logging.info('ANOSPP QC data import')

    hap_df = load_hap(args.haplotypes)
    
    run_id, comb_stats_df = load_comb_stats(args.stats)
    
    logging.info('plotting QC')
    
    if hap_df['target'].isin(CUTADAPT_TARGETS).all():
        anospp = True
        logging.info('only ANOSPP targets detected, plotting all ANOSPP QC plots')
    else:
        anospp = False
        logging.warning('non-ANOSPP targets detected, plotting only generic QC plots')

    fig, _ = plot_cov(comb_stats_df, run_id)
    fig.savefig(f'{args.outdir}/coverage.png')

    fig, _ = plot_target_balance(hap_df, run_id)
    fig.savefig(f'{args.outdir}/target_balance.png')

    # allele balance plotted for mosuqito targets only
    fig = plot_allele_balance(hap_df, run_id, anospp=anospp)
    fig.savefig(f'{args.outdir}/allele_balance.png')

    fig, _ = plot_well_balance(comb_stats_df, run_id)
    fig.savefig(f'{args.outdir}/well_balance.png')

    fig, _ = plot_sample_filtering(comb_stats_df, run_id, anospp=anospp)
    fig.savefig(f'{args.outdir}/filter_per_sample.png')

    # set of plots tweaked for anospp only 
    if anospp:
        # disabled as logistic fit does not work prior to filtering, 
        # post-filtering plot is generated as part of NN script
        # mosq_hap_df = hap_df[hap_df['target'].isin(MOSQ_TARGETS)]
        # fig, _, _, _ = plot_het_cov(mosq_hap_df, title='Raw mosquito', run_id=run_id)
        # fig.savefig(f'{args.outdir}/het_cov.png')

        fig, _ = plot_plate_stats(comb_stats_df, run_id, lims_plate=False)
        fig.savefig(f'{args.outdir}/plate_stats.png')

        fig, _ = plot_plate_summaries(comb_stats_df, run_id, lims_plate=False)
        fig.savefig(f'{args.outdir}/plate_summaries.png')

        fig, _ = plot_plate_summaries(comb_stats_df, run_id, lims_plate=True)
        fig.savefig(f'{args.outdir}/lims_plate_summaries.png')

        fig, _ = plot_sample_success(comb_stats_df, run_id, anospp=anospp)
        fig.savefig(f'{args.outdir}/sample_success.png')

        fig, _ = plot_plasm_balance(comb_stats_df, run_id)
        fig.savefig(f'{args.outdir}/plasm_balance.png')
    else:
        # enabled for non-ANOSPP data, untested
        fig, _, _, _ = plot_het_cov(hap_df, title='Total', run_id=run_id)
        fig.savefig(f'{args.outdir}/het_cov.png')

    if anospp:
        heatmap_cols = [
            'p1_reads',
            'p2_reads',
            'total_reads', 
            'target_reads',
            'overall_filter_rate',
            'raw_mosq_targets_recovered',
            'raw_multiallelic_mosq_targets',
            'unassigned_asvs'
            ]
    else:
        heatmap_cols = [
            'total_reads', 
            'target_reads',
            'overall_filter_rate',
            'targets_recovered',
            'unassigned_asvs'
            ]

    for col in heatmap_cols:
        for lims_plate in (True, False):
            # re-init heatmap args for each plot
            heatmap_kwargs = {
                    'annot':True,
                    'cmap':'coolwarm'
                }
            if col == 'raw_mosq_targets_recovered':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 62
            elif col == 'raw_multiallelic_mosq_targets':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = max(comb_stats_df[col])
            elif col == 'unassigned_asvs':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = max(comb_stats_df[col])
            elif col == 'overall_filter_rate':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 1
                heatmap_kwargs['fmt'] = '.2f'
            # read counts expected in all other cases
            else:
                # log-transform colour axis
                # vmin vmax set here
                heatmap_kwargs['norm'] = LogNorm(
                    vmin=0.1,
                    vmax=max(max(comb_stats_df[col]), 0.1))
                # auto-apply human_format to annot
                heatmap_kwargs['fmt'] = '' 
            

            fig, _ = plot_plate_heatmap(
                comb_stats_df,
                col=col,
                run_id=run_id,
                lims_plate=lims_plate,
                **heatmap_kwargs
                )
            if lims_plate:
                plate_hm_fn  = f'{args.outdir}/lims_plate_hm_{col}.png' 
            else:
                plate_hm_fn  = f'{args.outdir}/plate_hm_{col}.png' 
            fig.savefig(plate_hm_fn)
            plt.close(fig)

    logging.info('ANOSPP QC complete')


def main():
    
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', 
                        help='Haplotypes tsv file generated by prep script', 
                        required=True)
    parser.add_argument('-s', '--stats', 
                        help='Stats tsv file generated by prep script',
                        required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='qc')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    qc(args)


if __name__ == '__main__':
    main()