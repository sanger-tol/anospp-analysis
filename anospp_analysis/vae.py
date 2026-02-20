import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
import keras
from scipy.spatial import ConvexHull, Delaunay

from anospp_analysis.util import setup_logging
from anospp_analysis.nn import parse_seqids_series, construct_unique_kmer_table

#Variables
K = 8
LATENTDIM = 3
SEED = 374173
WIDTH = 128
DEPTH = 6
DISTRATIO = 7

def prep_reference_index(reference_path):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index {reference_path}')

    reference_path = reference_path.rstrip('/')

    reference_version = reference_path.split('/')[-1]

    assert re.match(r'^gcrefv\d', reference_version), f'{reference_version} not recognised as gc vae ref version'

    assert os.path.isdir(reference_path), \
        f'reference version {reference_version} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/selection_criteria.txt'), \
        f'reference version {reference_version} at {reference_path} does not contain required selection_criteria.txt file'
    selection_criteria_file = f'{reference_path}/selection_criteria.txt'

    assert os.path.isfile(f'{reference_path}/_weights.hdf5'), \
        f'reference version {reference_version} at {reference_path} does not contain required _weights.hdf5 file'
    vae_weights_file = f'{reference_path}/_weights.hdf5'

    assert os.path.isfile(f'{reference_path}/convex_hulls.tsv'), \
        f'reference version {reference_version} at {reference_path} does not contain required convex_hulls.tsv file'
    convex_hulls_df = pd.read_csv(f'{reference_path}/convex_hulls.tsv', sep='\t')

    if not os.path.isfile(f'{reference_path}/colors.tsv'):
        logging.warning('No colors defined for plotting.')
        colorsdict = dict()
    else:
        colors = pd.read_csv(f'{reference_path}/colors.tsv',sep='\t', index_col=0)
        colorsdict = dict(colors.color)

    if not os.path.isfile(f'{reference_path}/latent_coordinates.tsv'):
        logging.warning('No reference coordinates defined for plotting.')
        ref_coord = pd.DataFrame()
    else:
        ref_coord = pd.read_csv(f'{reference_path}/latent_coordinates.tsv',sep='\t')

    if os.path.isfile(f'{reference_path}/version.txt'):
        with open(f'{reference_path}/version.txt', 'r') as fn:
            for line in fn:
                version_name = line.strip()
    else:
        logging.warning(f'No version.txt file present for reference version {reference_version} at {reference_path}')
        version_name = 'unknown'
    if version_name != reference_version:
        logging.warning(f'Reference directory name {reference_version} does not match version.txt: {version_name}')

        
    return (selection_criteria_file, vae_weights_file, convex_hulls_df, colorsdict, ref_coord, version_name)

# def read_selection_criteria(selection_criteria_file, nn_stats_df, nn_hap_df):
    
#     level, sgp, n_targets = open(selection_criteria_file).read().split('\t')
#     return select_samples(nn_stats_df, nn_hap_df, level, sgp, int(n_targets))

def select_samples(selection_criteria_file, nn_stats_df, nn_hap_df):
    '''
    Select the samples meeting the criteria for VAE assignment
    Based on NN assignment and number of targets
    '''
    #identify samples meeting selection criteria
    selection_criteria = pd.read_csv(selection_criteria_file, sep='\t')
    vae_samples = []
    for _, r in selection_criteria.iterrows():
        crit_vae_samples = nn_stats_df.loc[
            (nn_stats_df[f'nn_{r.level}'] == r.sgp) & (nn_stats_df['mosq_targets_recovered'] >= r.n_targets), 
            'sample_id'
            ]
        vae_samples.append(crit_vae_samples)
    vae_samples = pd.concat(vae_samples)
    
    #subset haplotype df
    vae_hap_df = nn_hap_df.query('sample_id in @vae_samples')
    
    logging.info(f'selected {len(vae_samples)} samples to be run through VAE')

    return vae_samples, vae_hap_df

def prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids):
    '''
    Prepare k-mer table for a single sample
    '''
    #set up empty arrays
    total_targets = kmers_unique_seqs.shape[0]
    table = np.zeros((total_targets, kmers_unique_seqs.shape[2]), dtype='int')
    n_haps = np.zeros((total_targets), dtype='int')

    for _, row in parsed_seqids.iterrows():
        #only record the first two haplotypes for each target
        if n_haps[row.target] < 2:
            n_haps[row.target] += 1
            table[row.target,:] += kmers_unique_seqs[row.target, row.uidx, :]
    #double counts for homs
    for target in np.arange(total_targets):
        if n_haps[target] == 1:
            table[target,:] *= 2
    #sum over targets
    summed_table = np.sum(table, axis=0)

    return summed_table

def prep_kmers(vae_hap_df, vae_samples, k):
    '''
    Prepare k-mer table for the samples to be run through VAE
    '''
    #translate unique sequences to k-mers
    kmers_unique_seqs = construct_unique_kmer_table(vae_hap_df, k, source='vae samples')
    
    # logging.info('generating k-mer tables for selected samples')

    #set up k-mer table
    kmers_samples = np.zeros((len(vae_samples), 4**k))

    #fill in table for samples
    for n, sample in enumerate(vae_samples):
        parsed_seqids = parse_seqids_series(
            vae_hap_df.loc[vae_hap_df.sample_id == sample, 'seqid']
            )
        kmers_samples[n,:] = prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids)

    return(kmers_samples)

def latent_space_sampling(args):
    
    #Add noise to encoder output
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(
        shape=(keras.backend.shape(z_mean)[0], LATENTDIM),
        mean=0, 
        stddev=1., 
        seed=SEED
        )
    
    return z_mean + keras.backend.exp(z_log_var) * epsilon

def define_vae_input(k):
    input_seq = keras.Input(shape=(4**k,))

    return input_seq

def define_encoder(k):
    input_seq = define_vae_input(k)
    x = keras.layers.Dense(WIDTH, activation = 'elu')(input_seq)
    for i in range(DEPTH - 1):
        x = keras.layers.Dense(WIDTH, activation = 'elu')(x)
    z_mean = keras.layers.Dense(LATENTDIM)(x)
    z_log_var = keras.layers.Dense(LATENTDIM)(x)
    z = keras.layers.Lambda(
        latent_space_sampling, 
        output_shape=(LATENTDIM,),
        name = 'z'
        )([z_mean, z_log_var])
    encoder = keras.models.Model(input_seq, [z_mean, z_log_var, z], name = 'encoder')

    return encoder

def define_decoder(k):
    #Check whether you need the layer part here
    decoder_input = keras.layers.Input(shape=(LATENTDIM,), name='ls_sampling')
    x = keras.layers.Dense(WIDTH, activation='linear')(decoder_input)
    for i in range(DEPTH - 1):
        x = keras.layers.Dense(WIDTH, activation='elu')(x)
    output = keras.layers.Dense(4**k, activation='softplus')(x)
    decoder = keras.models.Model(decoder_input, output, name='decoder')

    return decoder

def define_vae(k):
    input_seq = define_vae_input(k)
    encoder = define_encoder(k)
    decoder = define_decoder(k)
    output_seq = decoder(encoder(input_seq)[2])
    vae = keras.models.Model(input_seq, output_seq, name='vae')

    return vae, encoder

def predict_latent_pos(kmer_table, vae_samples, k, vae_weights_file):

    '''
    Predict latent space of test samples based on reference database
    '''

    logging.info('predicting latent space of test samples based on reference database')

    vae, encoder = define_vae(k)

    vae.load_weights(vae_weights_file)
    predicted_latent_pos = encoder.predict(kmer_table, verbose=False)

    predicted_latent_pos_df = pd.DataFrame(
        index=vae_samples, 
        columns=['mean1', 'mean2', 'mean3', 'sd1', 'sd2', 'sd3']
        )
    for i in range(3):
        predicted_latent_pos_df[f'mean{i + 1}'] = predicted_latent_pos[0][:,i]
        predicted_latent_pos_df[f'sd{i + 1}'] = predicted_latent_pos[1][:,i]

    return predicted_latent_pos_df

def generate_convex_hulls(convex_hulls_df):
    '''
    Read in pre-computed points for convex hulls for each species
    '''
    logging.info('setting up convex hulls from reference database')
    hull_dict = dict()
    for species in convex_hulls_df.species.unique():
        pos = convex_hulls_df.loc[
            convex_hulls_df.species==species, 
            ['mean1', 'mean2', 'mean3']
            ].values
        hull = ConvexHull(pos)
        hull_dict[species] = (pos, hull)

    return hull_dict

def check_half_space(p, n, o):
    '''
    check in which half space separated by the plane
    defined by its normal n p lies
    centered on origin o
    '''
    hp = np.dot(p - o, n)
    
    return hp >= 0

def find_normal_edge_simplex(a, b, c):
    '''
    find the normal vector of the plane C
    spanned by the edge ab of the simplex abc
    and the normal vector to the simplex
    centered on a
    oriented s.t. positive points into the simplex
    '''
    #the normal vector of the plane of the simplex
    #centered on a
    z = np.cross(b - a, c - a)
    #normalised
    u = z / np.sqrt(np.dot(z, z))
    #the edge ab normalised and centered on a
    e = (b - a) / np.sqrt(np.dot(b - a, b - a))
    #the unit normal of plane C
    n = np.cross(u, e)
    
    return n, a

def check_edge_projection(p, v, w):
    '''
    check whether p lies above the edge vw
    '''
    #let vw be the normal vector defining plane U
    #through v
    e = w - v
    #length of vw
    le = np.sqrt(np.dot(e, e))
    #unit vector along the edge, centered on v
    n = e / le
    #project p along the edge vw
    q = np.dot(p - v, n)
    
    return q, le

def check_edge_partition(p, vertices):
    '''
    given that p lies in the part bordering edge vw
    determine whether it is closest to the edge 
    or one of the vertices
    '''
    v, w = vertices
    #get lenght of projection q of p along the edge
    #and the lenght l of the edge
    q, l = check_edge_projection(p, v, w)
    
    if q < 0:
        #vertex v is closest
        return compute_distance_to_vertex(p, v)
    elif q > l:
        #vertex w is closest
        return compute_distance_to_vertex(p, w)
    else:
        #edge vw is closest
        return compute_distance_to_edge(p, v, w)

def check_vertex_partition(p, v, verts):
    '''
    given that p lies in the part bordering vertex v
    determine whether it is closest to the vertex 
    or one of the edges vw or vu
    '''
    w1, w2 = verts
    #get lenght of projection q1 of p along the edge vw1
    #and the lenght l1 of the edge vw1
    q1, l1 = check_edge_projection(p, v, w1)
    if q1 > 0:
        #corner of more than 90 degrees
        if q1 > l1:
            #vertex w1 is closest
            return compute_distance_to_vertex(p, w1)
        else:
            #edge vw1 is closest
            return compute_distance_to_edge(p, v, w1)
    q2, l2 = check_edge_projection(p, v, w2)
    if q2 > 0:
        #corner of more than 90 degrees
        if q2 > l2:
            #vertex w2 is closest
            return compute_distance_to_vertex(p, w2)
        else:
            #edge vw2 is closest
            return compute_distance_to_edge(p, v, w2)
    else:
        #vertex v is closest
        return compute_distance_to_vertex(p, v)

def compute_distance_to_plane(p, vertices):
    '''
    compute distance point p to the plane defined 
    by the vertices of the simplex
    '''
    a, b, c = vertices
    #find a vector perpendicular to the simplex
    #centered on a
    z = np.cross(b - a, c - a)
    #unit vector perpendicular to the plane
    u = z / np.sqrt(np.dot(z, z))
    #distance of p to the plane
    h = np.abs(np.dot(p - a, u))
    
    return h

def compute_distance_to_vertex(p, v):
    '''
    compute distance of point p to vertex v
    '''
    d = np.sqrt(np.dot(p - v, p - v))
    
    return d

def compute_distance_to_edge(p, v, w):
    '''
    compute the distance of p to the edge vw
    '''
    #let vw be the normal vector defining plane U
    #through v
    e = w - v
    #let n be unit vector along the edge
    n = e / np.sqrt(np.dot(e, e))
    #let q be the projection of p along the edge vw
    q = np.dot(p - v, n) * n
    #distance is the length of the difference between p and q
    d = np.sqrt(np.dot(p - v - q, p - v - q))
    
    return d

def compute_distance_to_simplex(p, vertices):
    '''
    Compute the 3d distance of point p to the triangle
    defined by the vertices
    '''
    a, b, c = vertices
    
    #get normal vectors to the planes containing the 
    #edges of the triangles and perpendicular to the simplex
    Cn, Co = find_normal_edge_simplex(a, b, c)
    Bn, Bo = find_normal_edge_simplex(c, a, b)
    An, Ao = find_normal_edge_simplex(b, c, a)
    
    #for each of the edge planes, check whether p lies
    #in the same halfplane of the simplex
    Cs = check_half_space(p, Cn, Co)
    Bs = check_half_space(p, Bn, Bo)
    As = check_half_space(p, An, Ao)
    
    partition = np.array([As, Bs, Cs])
    
    if np.sum(partition) == 3:
        #p lies within the contours of the simplex
        return compute_distance_to_plane(p, vertices)
    elif np.sum(partition) == 2:
        #p lies in a part bordering an edge
        return check_edge_partition(p, vertices[partition])
    elif np.sum(partition) == 1:
        #p lies in a part bordering a vertex
        return check_vertex_partition(p, vertices[partition][0], vertices[~partition])
    else:
        logging.error(
            'Something thought to be impossible happened in the convex hull computation - '
            'tell Marilou to retake her linear algebra exam'
            )

def compute_distance_to_hull(hull, positions):
    '''
    compute the distance of all given positions
    to the given convex hull
    in 3d
    '''
    #array to store distances to all simplices
    dists = np.zeros((positions.shape[0], hull.simplices.shape[0]))
    for i, s in enumerate(hull.simplices):
        #compute distance to each simplex in turn
        dists[:, i] = np.array([compute_distance_to_simplex(p, hull.points[s]) for p in positions])
    #get the distance to the closest simplex for each point
    distances = dists.min(axis=1)

    return distances

def check_is_in_hull(hull_pos, positions):
    '''
    Check whether a set of positions lies inside the specified hull
    '''
    if not isinstance(hull_pos,Delaunay):
        hull = Delaunay(hull_pos)

    in_hull = hull.find_simplex(positions) >= 0

    return in_hull

def get_unassigned_samples(latent_positions_df):
    unassigned = latent_positions_df.loc[latent_positions_df.vae_species_call.isnull()].index
    n_unassigned = len(unassigned)
    return unassigned, n_unassigned

def generate_hull_dist_df(hull_dict, latent_positions_df, unassigned):

    dist_df = pd.DataFrame(index=unassigned)
    positions = latent_positions_df.loc[unassigned ,['mean1', 'mean2', 'mean3']].values
    for species in hull_dict.keys():
        dist_df[species] = compute_distance_to_hull(hull_dict[species][1], positions)
    return dist_df 

def get_closest_hulls(hull_dict, latent_positions_df, unassigned):

    dist_df = generate_hull_dist_df(hull_dict, latent_positions_df, unassigned)
    summary_dist_df = pd.DataFrame(index=dist_df.index)
    summary_dist_df['dist1'] = dist_df.min(axis=1)
    summary_dist_df['species1'] = dist_df.idxmin(axis=1)
    summary_dist_df['dist2'] = dist_df.apply(lambda x: x.sort_values().iloc[1], axis=1)
    summary_dist_df['species2'] = dist_df.apply(lambda x: x.sort_values().index[1], axis=1)
    return summary_dist_df, dist_df

def assign_gam_col_band(latent_positions_df, summary_dist_df):
    #Determine which samples are in gamcol band
    gamcol_band = summary_dist_df.loc[
        (summary_dist_df.species1.isin(['Anopheles_gambiae', 'Anopheles_coluzzii'])) & \
        (summary_dist_df.species2.isin(['Anopheles_gambiae', 'Anopheles_coluzzii'])) & \
        (summary_dist_df.dist2 < 14)
        ].copy()
    #Make assignments for the samples in gamcol band
    if gamcol_band.shape[0] > 0:
        gamcol_dict = dict(gamcol_band.apply(
            lambda row: 'Uncertain_' + row.species1.split('_')[1] + '_' + row.species2.split('_')[1],
            axis=1))
        latent_positions_df.loc[gamcol_band.index, 'vae_species_call'] = latent_positions_df.loc[\
            gamcol_band.index
            ].index.map(gamcol_dict)
    return latent_positions_df, gamcol_band.shape[0]

def assign_to_closest_hull(latent_positions_df, summary_dist_df, unassigned):
    '''
    Assign samples that are much closer to one hull than to all others to that hull
    Currently defined as a distance 7 times smaller than all others
    '''
    fuzzy_hulls = summary_dist_df.loc[
        (DISTRATIO*summary_dist_df.dist1 < summary_dist_df.dist2) & \
        summary_dist_df.index.isin(unassigned)
        ].copy()
    if fuzzy_hulls.shape[0] > 0:
        fuzzy_hulls_dict = dict(fuzzy_hulls.species1)
        latent_positions_df.loc[unassigned, 'vae_species_call'] = latent_positions_df.loc[
            unassigned
            ].index.map(fuzzy_hulls_dict)
    return latent_positions_df, fuzzy_hulls.shape[0]

def assign_to_multiple_hulls(latent_positions_df, dist_df, unassigned):
    '''
    Assign samples in between hulls to multiple hulls in order of closeness
    '''
    between_hulls = dist_df.loc[dist_df.index.isin(unassigned)].copy()
    between_hulls_dict = dict(between_hulls.apply(
        lambda row: 'Uncertain_'+'_'.join(
            label.split('_')[1] for label in row.sort_values().index[
                row.sort_values() < DISTRATIO * row.min()
                ]
            ),
        axis=1))
    latent_positions_df.loc[unassigned, 'vae_species_call'] = latent_positions_df.loc[
        unassigned
        ].index.map(between_hulls_dict)
    return latent_positions_df, between_hulls.shape[0]

def perform_convex_hull_assignments(hull_dict, latent_positions_df):
    '''
    Perform convex hull assignments based on latent space positions
    '''
    logging.info('performing convex hull assignment')
    positions = latent_positions_df[['mean1', 'mean2', 'mean3']].values
    
    #first check which samples fall inside convex hulls
    for label in hull_dict.keys():
        is_in_hull = check_is_in_hull(hull_dict[label][0], positions)
        latent_positions_df.loc[is_in_hull, 'vae_species_call'] = label
    
    #Record unassigned samples 
    unassigned, n_unassigned = get_unassigned_samples(latent_positions_df)
    logging.info(
        f'{latent_positions_df.shape[0] - n_unassigned} samples fall inside convex hulls; '
        f'{n_unassigned} samples still to be assigned'
        )
    
    #for the unassigned samples, get distances to two closest hulls
    if n_unassigned > 0:
        summary_dist_df, dist_df = get_closest_hulls(hull_dict, latent_positions_df, unassigned)
        latent_positions_df, n_newly_assigned = assign_gam_col_band(
            latent_positions_df, 
            summary_dist_df
            )
        unassigned, n_unassigned = get_unassigned_samples(latent_positions_df)
        logging.info(
            f'{n_newly_assigned} samples assigned to uncertain_gambiae_coluzzii or uncertain_coluzzii_gambiae; '
            f'{n_unassigned} samples still to be assigned'
            )
    if n_unassigned > 0:
        latent_positions_df, n_newly_assigned = assign_to_closest_hull(
            latent_positions_df, 
            summary_dist_df, 
            unassigned
            )
        unassigned, n_unassigned = get_unassigned_samples(latent_positions_df)
        logging.info(
            f'{n_newly_assigned} additional samples assigned to a single species; '
            f'{n_unassigned} samples still to be assigned')
    if n_unassigned > 0:
        latent_positions_df, n_newly_assigned = assign_to_multiple_hulls(
            latent_positions_df,
            dist_df, 
            unassigned
            )
        logging.info(f'{n_newly_assigned} samples assigned to multiple hulls.')
    logging.info(f'finished assigning {latent_positions_df.shape[0]} samples')

    return latent_positions_df

def plot_vae_assignments(ch_assignments, ref_coordinates, colordict, run_id):

    logging.info('generating VAE plots')
    #get assignment categories
    single, double, multi = [], [], []
    species_dict = dict(ch_assignments.groupby('vae_species_call').size())
    for key in species_dict.keys():
        #for single species assignment
        if key.startswith('Anopheles_'):
            single.append(key)
        #for uncertain between two species
        elif len(key.split('_')) == 3:
            double.append(key)
        #for uncertain between more than two species
        else:
            multi.append(key)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    fig.suptitle(f'An. gambiae complex VAE assignment for run {run_id}')

    #Plot reference dataset
    ref_coordinates['color'] = ref_coordinates.vae_species_call.map(colordict)
    ax.scatter(
        ref_coordinates.mean1, 
        ref_coordinates.mean3, 
        c=ref_coordinates.color.values, 
        alpha=.6, 
        zorder=1
        )
    
    #Plot samples assigned to single species
    if len(single) > 0:
        for species in single:
            sub = ch_assignments.query('vae_species_call == @species')
            if species == 'Anopheles_bwambae-fontenillei':
                label = f'Anopheles_bwambae-\nfontenillei: {species_dict[species]}'
            else:
                label = f'{species}: {species_dict[species]}'
            ax.scatter(
                sub.mean1, 
                sub.mean3, 
                color=colordict[species], 
                marker='^', 
                edgecolor='k', 
                label=label, 
                zorder=2
                )

    #Plot samples assigned to two species
    if len(double)>0:    
        for species in double:
            sub = ch_assignments.query('vae_species_call == @species')
            _, sp1, sp2 = species.split('_')
            #multiple lines on the legend
            if sp1 == 'bwambae-fontenillei':
                label = f'Uncertain_bwambae-\nfontenillei_{sp2}: {species_dict[species]}'
            else:
                label = f'Uncertain_{sp1}_\n{sp2}: {species_dict[species]}'
            ax.scatter(
                sub.mean1, 
                sub.mean3, 
                marker='s', 
                edgecolor=colordict[f'Anopheles_{sp1}'], 
                facecolor=colordict[f'Anopheles_{sp2}'], 
                linewidth=1.5, 
                zorder=2, 
                label=label
                )

    #Plot remaining samples
    if len(multi)>0:    
        sub = ch_assignments.query('vae_species_call in @multi')
        ax.scatter(
            sub.mean1, 
            sub.mean3,
            color='k', 
            marker='s', 
            zorder=3, 
            label=f'other: {sub.shape[0]}'
            )

    ax.set_xlabel('LS1')
    ax.set_ylabel('LS3')
        
    legend = ax.legend(loc='lower right', prop={'size': 9})

    return fig, ax

def generate_summary(ch_assignments, nn_stats_df, version_name):
    summary = [
        f'Convex hull assignment assignment using reference version {version_name}',
        f'On run containing {nn_stats_df.sample_id.nunique()} samples',
        f'{ch_assignments.index.nunique()} samples met the VAE selection criteria',
        f'Samples are assigned to the following labels:'
    ]
    assignment_dict = dict(ch_assignments.groupby('vae_species_call').size())
    for key in assignment_dict.keys():
        summary.append(f'{key}:\t {assignment_dict[key]}')

    return '\n'.join(summary)


def vae(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP VAE data import started')

    nn_hap_df = pd.read_csv(args.nn_haplotypes, sep='\t')
    nn_stats_df = pd.read_csv(args.nn_manifest, sep='\t')
    run_id = nn_stats_df['run_id'].iloc[0]

    selection_criteria_file, vae_weights_file, convex_hulls_df, colordict, ref_coordinates, \
        version_name = prep_reference_index(args.reference_path)
    vae_samples, vae_hap_df = select_samples(
        selection_criteria_file,
        nn_stats_df,
        nn_hap_df
        )
    if len(vae_samples) == 0:
        logging.info('No samples to be run through VAE - skipping to finalising assignments')
        ch_assignment_df = pd.DataFrame(columns = [
            'mean1', 'mean2', 'mean3', 'sd1', 'sd2', 'sd3', 'vae_species_call'
            ])
        ch_assignment_df.index.name = 'sample_id'

    else:
        kmer_table = prep_kmers(vae_hap_df, vae_samples, K)
        latent_positions_df = predict_latent_pos(kmer_table, vae_samples, K, vae_weights_file)
        hull_dict = generate_convex_hulls(convex_hulls_df)
        ch_assignment_df = perform_convex_hull_assignments(hull_dict, latent_positions_df)
        if not args.no_plotting:
            fig, _ = plot_vae_assignments(ch_assignment_df, ref_coordinates, colordict, run_id)
            fig.savefig(f'{args.outdir}/vae_assignment.png')

    ch_assignment_df['vae_ref'] = version_name
    logging.warning('Changing VAE species predictions prefix from "Anopheles_" to "An_"')
    ch_assignment_df['vae_species_call'] = ch_assignment_df['vae_species_call'] \
        .str.replace('^Anopheles_','An_', regex=True)
    ch_assignment_df.to_csv(f'{args.outdir}/vae_assignment.tsv', sep='\t')

    summary_text = generate_summary(ch_assignment_df, nn_stats_df, version_name)
    logging.info(f'writing summary file to {args.outdir}')
    with open(f'{args.outdir}/summary.txt', 'w') as fn:
        fn.write(summary_text)

    logging.info('ANOSPP VAE complete')

    
def main():
    
    parser = argparse.ArgumentParser('VAE assignment for samples in Anopheles gambiae complex')
    parser.add_argument('-a', '--nn_haplotypes',
                        help=f'nn_hap_summary.tsv generated by anospp-nn with k={K}',
                        required=True)
    parser.add_argument('-m', '--nn_manifest',
                        help=f'nn_assignment.tsv file generated by anospp-nn with k={K}',
                        required=True)
    parser.add_argument('-r', '--reference_path', 
                        help='Path to vae reference index directory',
                        required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: vae', default='vae')
    parser.add_argument('--no_plotting', 
                        help='Do not generate plots. Default: False',
                        action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    vae(args)


if __name__ == '__main__':
    main()