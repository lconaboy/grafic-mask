import numpy as np
from grid import Hierarchy

levelmin = 8
levelmax = 9
# path = '/home/lc589/projects/lg/test_g9p/'
ix = 225
iy = 225
iz = 225
nx = 60
ny = 60
nz = 60
pad = 4
ii = np.array([ix, iy, iz])
nn = np.array([nx, ny, nz])
h = Hierarchy(levelmin=levelmin, levelmax=levelmax,
              ii=ii, nn=nn, pad=pad)

# We build the refinement mask from the finest level down to the
# coarsest level, then do a pass from coarse to fine to check that
# the mask is consistent. Starting with a general cubic mask.
