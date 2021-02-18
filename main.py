import numpy as np


class Refmap:
    def __init__(self, nn, ii, l, v=True):
        """An individual level's refinement map, accessed through
        refmap.

        :param nn: (array, ints) extent of the level
        :param ii: (array, ints) origin of the level
        :param l: (int) ilevel
        :param v: (bool) verbose

        :returns:

        :rtype:
        """
        self.nn = nn
        self.ii = ii
        self.l = l
        self.v = v

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
        :param pad: (int) padding of each level, in ilevel cells
        :param v: (bool) verbose

        :returns:

        :rtype:
        """
        self.lmin = levelmin
        self.lmax = levelmax
        self.ii_fine = ii  # start with finest
        self.nn_fine = nn  # start with finest
        self.pad = 4  # per side
        self.v = v

        self.set_hierarchy()

    def set_hierarchy(self):
        """Builds the hierarchy of Refmaps, which are stored in a list
        accessed through h.  The elements of the list are Refmaps
        ordered from levelmin to levelmax.

        :returns:

        :rtype:
        """
        self.h = []

        # Do down to, but not including levelmin
        for il in range(self.lmax, self.lmin, -1):
            # Prepend each Refmap, so that they are ordered from
            # smallest to largest
            self.h.insert(0, Refmap(self.nn_fine, self.ii_fine, il,
                                    v=self.v))

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
        self.h.insert(0, Refmap(self.nn_fine, self.ii_fine, self.lmin,
                                v=self.v))

        
            
