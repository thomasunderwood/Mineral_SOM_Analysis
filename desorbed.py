import numpy as np
import MDAnalysis as mda

# load universe
u       = mda.Universe('./topol.tpr', './traj_comp.xtc')

water = u.select_atoms('resname SOL')
ions = u.select_atoms('resname Ca Na K')
SOM = u.select_atoms('resname PROA PRAC AMOP AMOL 3CB 3CPY FBIB BF7 BCA IP_2 C37 C3C PACP ZTHP FEOZ QINL GLYCYS EPRE MBUT MPRO')
mineral = u.select_atoms('resname MMT MICA GOE')

import MDAnalysis.transformations as trans
transforms = [trans.unwrap(mineral),
              trans.center_in_box(mineral, wrap=True),
              trans.wrap(u.select_atoms('all'))]
u.trajectory.add_transformations(*transforms)

box_z = u.dimensions[2]
midpoint_z = box_z/2

phase_above = u.select_atoms('not resname SOL MMT MICA GOE and prop z > {}'.format(midpoint_z), updating=True)
phase_below = u.select_atoms('not resname SOL MMT MICA GOE and prop z < {}'.format(midpoint_z), updating=True)

from collections import Counter
from MDAnalysis.lib.log import ProgressBar
desorbed = Counter()
for ts in ProgressBar(u.trajectory):
    desorbed += (Counter(phase_below.residues.resnames))

print(desorbed.items())

import pandas as pd
# Save all arrays...
pd.DataFrame(desorbed.items(), columns=['ResNames', '#Desorbed']).to_csv('desorbed.csv', index=False)