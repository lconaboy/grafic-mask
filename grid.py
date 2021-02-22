import numpy as np


class Refmap:
    def __init__(self, nn, ii, io, l, v=True):
        """An individual level's refinement map, accessed through
        refmap.

        :param nn: (array, ints) extent of the level, in this level's
            cells
        :param ii: (array, ints) offset of the origin of the level, in
            this level's cells
        :param io: (array, ints) offset of the origin of the level
            relative to the next coarser level, in the coarser level's
            cells
        :param l: (int) ilevel
        :param v: (bool) verbose

        :returns:

        :rtype:
        """
        self.nn = nn
        self.ii = ii
        self.io = io
        self.l = l
        self.v = v

        assert(self.l > 0), self.l  # should be working on a level > 0
        assert(np.all(self.nn > 0)), self.nn  # should be some grids
        assert(np.all(self.ii >= 0)), self.ii  # coords should be positive
        assert(np.all(self.io >= 0)), self.io  # offsets should be positive

        # Initialise empty dict for the cosmo
        self.cosmo = {}
        
        # Check the origin
        self.set_origin()

        # Build the empty refmap
        self.set_refmap()


    def set_origin(self):
        """Updates the origin and extent of the refmap so that it
        aligns with the next-coarsest grid.

        :returns:

        :rtype:
        """
        # Check that the supplied origin aligns with the coarser
        # grids. If they don't, shift the origin to the left.
        _ii = self.ii - np.mod(self.ii, 2)

        # Now check the right side. This should also align with the
        # coarser grid.
        _jj = _ii + self.nn
        _jj = _jj + np.mod(_jj, 2)
        _nn = _jj - _ii

        # These should all be even now, so let's check that
        assert(np.all(np.mod(_ii, 2) == 0)), _ii
        assert(np.all(np.mod(_nn, 2) == 0)), _nn
        
        self.ii = _ii
        self.nn = _nn

        
    def set_refmap(self):
        """Sets the base refmap on to which the refinement map will be
        built.

        :returns:

        :rtype:
        """
        
        self.refmap = np.zeros(shape=(self.nn[0], self.nn[1], self.nn[2]),
                               dtype=np.float32)


    def set_cosmo(self, cosmo):
        self.cosmo = cosmo
        

class Hierarchy:
    def __init__(self, levelmin, levelmax, ii, nn, pad=4, v=True):
        """Class for storing a hierarchy of nested grids (Refmaps).
        The Refmaps are accessed through h.

        :param levelmin: (int) smallest level in the ICs
        :param levelmax: (int) largest level in the ICs
        :param ii: (array, ints) origin of the finest grid, in fine
            grid cells
        :param nn: (array, ints) extent of the finest grid, in fine
            grid cells
        :param pad: (int) padding of each level on each side, in
            ilevel cells
        :param v: (bool) verbose

        :returns:

        :rtype:
        """
        self.lmin = levelmin
        self.lmax = levelmax
        self.nl = self.lmax - self.lmin + 1
        self.ii_fine = ii  # start with finest
        self.nn_fine = nn  # start with finest
        self.pad = pad     # per side
        self.v = v

        # Build iterators for the levels
        self.il_it = range(self.nl)  # forward sweep
        self.il_itr = range(self.nl-1, -1, -1)  # reverse sweep

        
        self.set_hierarchy()

    def set_hierarchy(self):
        """Builds the hierarchy of Refmaps, which are stored in a list
        accessed through h.  The elements of the list are Refmaps
        ordered from levelmin to levelmax.

        :returns:

        :rtype:
        """
        self.h = []

        # Initially set the relative offsets to some dummy value,
        # we'll go back and update these after building the hierarchy
        _io_coarse = np.array([0, 0, 0])
        
        # Do down to, but not including levelmin
        for ll in range(self.lmax, self.lmin, -1):
            # Prepend each Refmap, so that they are ordered from
            # smallest to largest
            self.h.insert(0, Refmap(nn=self.nn_fine, ii=self.ii_fine,
                                    io=_io_coarse, l=ll, v=self.v))

            # Update nn_fine and ii_fine for the next level
            _nn_fine = self.nn_fine // 2
            _nn_fine = _nn_fine + (2 * self.pad)

            _ii_fine = self.ii_fine // 2
            _ii_fine = _ii_fine - self.pad

            self.nn_fine = _nn_fine
            self.ii_fine = _ii_fine

        # Now do levelmin
        _nn_fine = 2 ** self.lmin
        self.nn_fine = np.array([_nn_fine, _nn_fine, _nn_fine])
        self.ii_fine = np.array([0, 0, 0])
        self.h.insert(0, Refmap(nn=self.nn_fine, ii=self.ii_fine,
                                io=_io_coarse, l=self.lmin, v=self.v))

        # Now do a forward sweep to calculate the relative offsets
        # between levels, levelmin has a relative offset of zero, so
        # we can start from lmin+1
        for il in range(1, self.nl):
            # level il absolute offset in coarse cells - level il-1
            # absolute offset in coarse cells gives relative offset
            # between il and il-1 in coarse (il-1) cells
            _io_coarse = (self.h[il].ii // 2) - self.h[il-1].ii

            # Update the relative offset
            self.h[il].io = _io_coarse
