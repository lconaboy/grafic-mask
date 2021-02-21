import os
import numpy as np
from scipy.io import FortranFile

def _write(fn, h, x):
    
    with open(fn, 'w+') as f:
        # Write the header
        _write_header(f, h)
        
        # Loop through x and write the transposed slabs
        for k in range(x.shape[2]):
            # f.write_record(x[:, :, k].T)
            _write_slab(f, x[:, :, k].T)


def _set_fn(level, field):
    
    path = 'level_{0:03d}'.format(level)
    if not os.path.isdir(path):
        os.mkdir(path)
        
    path += '/ic_{0}'.format(field)
    if os.path.isfile(path):
        print('---- [warning] overwriting', path)

    return path


def _write_header(f, h):
    n1 = h.nn[0]
    n2 = h.nn[1]
    n3 = h.nn[2]
    
    dx = h.cosmo['dx']

    x1o = h.io[0] * dx
    x2o = h.io[1] * dx
    x3o = h.io[2] * dx

    astart = h.cosmo['astart']
    omegam = h.cosmo['omegam']
    omegav = h.cosmo['omegav']
    h0 = h.cosmo['h0']

    nb = np.array([44], dtype=np.int32)
    
    nb.tofile(f)  # header bytes
    np.array([n1, n2, n3], dtype=np.int32).tofile(f)  # ints
    np.array([dx, x1o, x2o, x3o, astart, omegam, omegav, h0],
             dtype=np.float32).tofile(f)  # reals
    nb.tofile(f)  # header bytes


def _write_slab(f, s):
    nb = np.array([s.shape[0] * s.shape[1] * 4], dtype=np.int32)

    nb.tofile(f)
    s.tofile(f)
    nb.tofile(f)
    
    
def write_field(field, h, x, level):
    fn = _set_fn(level, field)
    _write(fn, h, x)
