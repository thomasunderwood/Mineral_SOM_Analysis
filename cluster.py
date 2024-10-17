import MDAnalysis as mda
import numpy as np
from itertools import combinations
from MDAnalysis.lib.distances import distance_array
from collections import defaultdict, Counter

cutoff = 2.82

# load universe
u       = mda.Universe('./topol.tpr', './traj_comp.xtc')
SOM     = u.select_atoms('resname PROA PRAC AMOP AMOL 3CB 3CPY FBIB BF7 BCA IP_2 C37 C3C PACP ZTHP FEOZ QINL GLYCYS EPRE MBUT MPRO')

ions    = u.select_atoms('resname Ca Na K')
ion = ions.residues.resnames
ion = np.unique(ion)[0]
if ion == 'Na': 
    ion_cutoff = 4.5
elif ion == 'Ca': 
    ion_cutoff = 5
else: print('error in ion cutoff')
print('ion_cutoff =', ion_cutoff)

cluster_data = np.empty((u.trajectory.n_frames, SOM.residues.resnames.shape[0]), dtype=np.complex_) * np.NaN

from MDAnalysis.lib.log import ProgressBar
for ts in ProgressBar(u.trajectory):
    water   = u.select_atoms('resname SOL')
    ions    = u.select_atoms('resname Ca Na K')
    SOM     = u.select_atoms('resname PROA PRAC AMOP AMOL 3CB 3CPY FBIB BF7 BCA IP_2 C37 C3C PACP ZTHP FEOZ QINL GLYCYS EPRE MBUT MPRO')
    mineral = u.select_atoms('resname MMT MICA GOE')
    
    # Centre the mineral for analysis
    mineral_center = mineral.center_of_mass(wrap=True)
    dim = ts.triclinic_dimensions
    box_center = np.sum(dim, axis=0) / 2
    u.atoms.translate(box_center - mineral_center)
    u.atoms.wrap(compound='residues')
    
    # Filter out the organics below the mineral
    lowest_mineral_z = min(mineral.atoms.positions[:, 2])
    # print('prop z < {}'.format(lowest_mineral_z))
    SOM_below_mineral = SOM.select_atoms('prop z < {}'.format(lowest_mineral_z), updating=True)
    SOM_above_mineral = SOM - SOM_below_mineral.residues.atoms
    
    # Calculate ions sorbed to mineral surface
    sorbed_ions = u.atoms[[]]
    for residue in ions.residues:
        distance_matrix = distance_array(residue.atoms, mineral.atoms, box=u.dimensions)
        if (distance_matrix < ion_cutoff).any(): sorbed_ions += residue.atoms
            
    # Calculate OM directly sorbed to mineral surface
    directly_sorbed_OM = u.atoms[[]]
    for residue in SOM_above_mineral.residues:
        distance_matrix = distance_array(residue.atoms, mineral.atoms, box=u.dimensions)
        if (distance_matrix < cutoff).any(): directly_sorbed_OM += residue.atoms
    
    # Calculate OM sorbed to mineral surface through a cation bridge
    cation_bridged_OM = u.atoms[[]]
    for residue_A in SOM_above_mineral.residues:
        for residue_B in sorbed_ions.residues:
            distance_matrix = distance_array(residue_A.atoms, residue_B.atoms, box=u.dimensions)
            if (distance_matrix < cutoff).any(): cation_bridged_OM += residue_A.atoms         
    cation_bridged_OM -= directly_sorbed_OM
    sorbed_OM = cation_bridged_OM + directly_sorbed_OM

    # Should I include the ions in the clustering analysis here?
    SOM_ions = SOM_above_mineral + ions
    
    pairs = []
    comb = combinations(SOM_ions.split('residue'), 2)
    # check for neighbours between unique pairs
    for pair in list(comb):
        residue_A, residue_B = pair
        distance_matrix = distance_array(residue_A, residue_B, box=u.dimensions)
        if (distance_matrix < cutoff).any(): pairs.append([(residue_A.resnums[0]), (residue_B.resnums[0])])
            
    # combine two-pair clusters to larger cluster network
    def connected_components(lists):
        neighbors = defaultdict(set)
        seen = set()
        for each in lists:
            for item in each:
                neighbors[item].update(each)
        def component(node, neighbors=neighbors, seen=seen, see=seen.add):
            nodes = set([node])
            next_node = nodes.pop
            while nodes:
                node = next_node()
                see(node)
                nodes |= neighbors[node] - seen
                yield node
        for node in neighbors:
            if node not in seen:
                yield sorted(component(node))
    
    cluster_with_ions = list(connected_components(pairs))
    
    # Prune ions from clusters
    clusters = []
    for cluster in cluster_with_ions:
        clusters.append([i for i in cluster if i < 85]) # 84 residues of OM
    clusters = [i for i in clusters if len(i) > 1]
    
    # Filter clusters by size
    clusters.sort(key=len, reverse=True)
    
    # Find if clusters are attached to mineral
    for count, cluster in enumerate(clusters, start=1):
        # Check if any residue in the cluster is part of the sorbed OM phase
        # NB 'i' is the residue number of the organic in the entire system
        # I believe i need to subtract 2 here. 1 from 0-indexing in python, 1 as the zeroth residue is the mineral...
        if any([i for i in cluster if i in sorbed_OM.residues.resnums]):
#             print(count, 'bang - sorbed cluster', np.array(cluster))
            cluster_data[ts.frame, np.array(cluster) - 2] = count
        else: 
#             print(count, 'no bang - desorbed cluster', cluster)
            cluster_data[ts.frame, np.array(cluster) - 2] = -count
    
    # Find aggregate bound SOM...
    for resid in np.arange(84):
        if (np.isnan(cluster_data[ts.frame, resid])) & (resid in (sorbed_OM.residues.resnums - 2)): 
#             print(resid)
            cluster_data[ts.frame, resid] = 0
    
    # Tag sorbed_OM as imaginary
    for resid in np.arange(84):
        if (resid in (sorbed_OM.residues.resnums - 2)): 
            cluster_data[ts.frame, resid] = cluster_data[ts.frame, resid] * 1j
    
# Save to array
print(cluster_data)
np.save('./cluster_data.npy', cluster_data)
