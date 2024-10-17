import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pytim as pytim
import MDAnalysis.transformations as trans
import pandas as pd

from tess import Container
from MDAnalysis.analysis import distances

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def TwoD_area(cntr):
    area=0
    # for index, element in enumerate(cntr):
    vertices = np.array(cntr.vertices())[:, 0:2] # Extract only {x,y} points for vertices
    vertices = np.unique(vertices, axis=0) # Trim duplicates
    y = vertices[:, 1]
    x = vertices[:, 0]
    order = np.argsort(np.arctan2(y - y.mean(), x - x.mean())) # Order vertices in clockwise manner
    area += PolyArea(x[order], y[order]) # Calculate their area
    return area
    
def TwoD_tesselation(frame, area_dictionary):
    box_x, box_y, box_z = frame.dimensions[0:3]
    box_z = 0.00001
    area = 0
    atom_positions = frame.atoms.positions[:, :]
    atom_positions[:, 2] = 0 
    
    try:
        cntr = Container(points=atom_positions, limits=np.array([box_x, box_y, box_z]), periodic=True)
    except:
        return np.nan, np.nan, np.nan

    for i in cntr:
        atomtype = frame[i.id].type
        area = TwoD_area(i)
        area_dictionary[atomtype] += area
        # print(frame[i.id].type, TwoD_area(i))
    return
    

u       = mda.Universe('./topol.tpr', './traj_comp.xtc')

water   = u.select_atoms('resname SOL')
ions    = u.select_atoms('resname Ca Na K')
SOM     = u.select_atoms('resname PROA PRAC AMOP AMOL 3CB 3CPY FBIB BF7 BCA IP_2 C37 C3C PACP ZTHP FEOZ QINL GLYCYS EPRE MBUT MPRO')
mineral = u.select_atoms('resname MMT MICA GOE')
# ao      = u.select_atoms('type ao')
# mgo     = u.select_atoms('type mgo')

transforms = [trans.unwrap(mineral),
              trans.center_in_box(mineral, wrap=True),
              trans.wrap(u.select_atoms('all'))]
u.trajectory.add_transformations(*transforms)

box_z = u.dimensions[2]
midpoint_z = box_z/2

# phase_above = u.select_atoms('not resname MMT MICA GOE and prop z > {}'.format(midpoint_z))
# phase_above = u.select_atoms('(not resname MMT MICA GOE) and (not element H) and (prop z > {})'.format(midpoint_z))
phase_above = u.select_atoms('(not resname MMT MICA GOE) and (not element H) and not (resname SOL and around 3.1 element Fe) and (prop z > {})'.format(midpoint_z))
# goethite_water = u.select_atoms('resname SOL and around 2.5 element Fe'))


""" Van der Waals radii in [A] taken from:
A cartography of the van der Waals territories
S. Alvarez, Dalton Trans., 2013, 42, 8617-8636
DOI: 10.1039/C3DT50599E
"""
vdw_radii = {
    'Si' : 2.19,
    'Al' : 2.25,
    'Mg' : 2.51,
    'O'  : 1.5,
    'H'  : 1.2,
    'C'  : 1.77,
    'N'  : 1.66,
    'S'  : 1.89,
    'P'  : 1.9,
    'Na' : 2.5, 
    'Ca' : 2.62,
}

interface = pytim.ITIM(u, 
                       group=phase_above, 
                       alpha=1.5, 
                       max_layers=4, 
                       molecular=False,
                       normal='z',
                       radii_dict=vdw_radii,
                       # centered='middle',
                       cluster_cut=None,
                       autoassign=True,
                       info=True,
                       warnings=True)

# Setup dictionary for 2D area dictionaries
atomtypes = np.unique(u.atoms.types)
n_atomtypes = len(atomtypes)

layer0_area = {atomtype: 0 for atomtype in atomtypes}
layer1_area = {atomtype: 0 for atomtype in atomtypes}
layer2_area = {atomtype: 0 for atomtype in atomtypes}
layer3_area = {atomtype: 0 for atomtype in atomtypes}

from MDAnalysis.lib.log import ProgressBar
for ts in ProgressBar(u.trajectory):
    i = ts.frame 
    # For each layer
        # Extract {x,y} coords
        # 2D voronoi tesselation
        # Extract area of each phase
        # PDF for each species with Octohedral atoms in 2D
    layer0 = interface.layers[1,:][0]
    layer1 = interface.layers[1,:][1]
    layer2 = interface.layers[1,:][2]
    layer3 = interface.layers[1,:][3]
    TwoD_tesselation(layer0, layer0_area)
    TwoD_tesselation(layer1, layer1_area)
    TwoD_tesselation(layer2, layer2_area)
    TwoD_tesselation(layer3, layer3_area)

print(layer0_area)
print(layer1_area)
print(layer2_area)
print(layer3_area)

# Save all arrays...
pd.DataFrame(layer0_area.items(), columns=['AtomTypes', 'Planar-Area']).to_csv('layer0_area_noH.csv', index=False)
pd.DataFrame(layer1_area.items(), columns=['AtomTypes', 'Planar-Area']).to_csv('layer1_area_noH.csv', index=False)
pd.DataFrame(layer2_area.items(), columns=['AtomTypes', 'Planar-Area']).to_csv('layer2_area_noH.csv', index=False)
pd.DataFrame(layer3_area.items(), columns=['AtomTypes', 'Planar-Area']).to_csv('layer3_area_noH.csv', index=False)
