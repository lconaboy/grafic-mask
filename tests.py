import numpy as np


def print_info(function):
    def wrapper():
        fn = function.__name__
        print('---- testing {0}'.format(fn))
        function()
        print('---- passed {0}'.format(fn))

    return wrapper


@print_info
def build_refmap():
    from grid import Refmap
    nn = np.array([64, 64, 64])
    ii = np.array([0, 0, 0])
    io = np.array([0, 0, 0])
    l = 6
    
    m = Refmap(nn=nn, ii=ii, io=io, l=l)

    assert(np.all(m.nn == nn))
    assert(np.all(m.refmap.shape == nn))


@print_info
def shift_origin():
    from grid import Refmap
    nn = np.array([64, 64, 64])
    ii = np.array([0, 0, 0])
    io = np.array([0, 0, 0])
    l = 6
    m = Refmap(nn=nn, ii=ii, io=io, l=l)

    # Should stay the same
    assert(np.all(m.ii == ii)), m.ii

    ii = np.array([12, 5, 37])
    m = Refmap(nn=nn, ii=ii, io=io, l=l)

    # Should change
    assert(np.all(m.ii == np.array([12, 4, 36]))), m.ii


@print_info
def build_hierarchy():
    from grid import Hierarchy
    lmin = 5
    lmax = 7
    ii = np.array([48, 48, 48])  # Should be even
    nn = np.array([8, 8, 8])  # Should be divisible to an even number lmin - lmax times
    pad = 4
    
    h = Hierarchy(lmin, lmax, ii, nn)

    # Should build the right number of refmaps
    assert(len(h.h) == (lmax - lmin + 1)), len(h.h)

    # The refmaps should be increasing in level
    for i in range(lmax - lmin + 1):
        assert(np.all(h.h[i].l == (lmin + i))), h.h[i].l

    # Check the size of the refmaps is decreasing accordingly
    _nn = nn
    for i in range(lmax - lmin, 0, -1):
        assert(np.all(h.h[i].nn == _nn)), h.h[i].nn
        _nn = (_nn // 2) + (2 * pad)


def update_rel_offset():
    pass


@print_info
def write_fields():
    from grid import Hierarchy
    from grafic_io import write_field
    lmin = 5
    lmax = 7
    ii = np.array([48, 48, 48])  # Should be even
    nn = np.array([8, 8, 8])  # Should be divisible to an even number lmin - lmax times
    pad = 4
    
    h = Hierarchy(lmin, lmax, ii, nn)
    for level in range(lmin, lmax+1):
        cosmo = {'h0': 70, 'omegam': 0.3, 'omegav': 0.7, 'astart': 0.01,
                 'dx':0.5*(level - lmin)}
        h.h[level-lmin].set_cosmo(cosmo)
        
        for field in ['deltab', 'poscx', 'poscy', 'poscz',
                      'velcx', 'velcy', 'velcz', 'refmap']:
            write_field(field, h.h[level-lmin], h.h[level-lmin].refmap, level)


@print_info
def run_iters():
    from grid import Hierarchy
    
    lmin = 5
    lmax = 7
    ii = np.array([48, 48, 48])  # Should be even
    nn = np.array([8, 8, 8])  # Should be divisible to an even number lmin - lmax times
    pad = 4

    nl = lmax - lmin + 1
    
    h = Hierarchy(lmin, lmax, ii, nn)

    for i, il in enumerate(h.il_it):
        assert(il == i)

    for i, il in enumerate(h.il_itr):
        assert(il == nl-1-i)


            
if __name__ == '__main__':
    print('-- running tests')
    
    build_refmap()
    shift_origin()
    build_hierarchy()
    write_fields()
    run_iters()
