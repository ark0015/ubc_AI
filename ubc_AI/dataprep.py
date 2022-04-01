#!/usr/bin/env python
from ubc_AI.prepfold import pfd
import numpy as np
import scipy
import sys, os


if __name__ == "__main__":
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    if not (f1.endswith("pfd")):
        print("file name %s not end with pfd " % (f1))
        sys.exit(1)

    pfdfile = pfd(f1)
    pfdfile.dedisperse()
    profs = pfdfile.profs
    pshape = profs.shape
    x, y, z = profs.shape
    data = profs.reshape((-1, 1))
    del pfdfile

    mean = np.mean(data)
    var = np.std(data)
    data = (data - mean) / var
    profs = data.reshape(pshape)
    X, Y, Z = scipy.ogrid[0:1:x, 0:1:y, 0:1:z]
    coords = scipy.array([X, Y, Z])
    coeffs = scipy.ndimage.spline_filter(profs)
    X, Y, Z = scipy.mgrid[0:1:8j, 0:1:8j, 0:1:8j]
    coords = scipy.array([X, Y, Z])
    newf = scipy.ndimage.map_coordinates(coeffs, coords, prefilter=False)

    np.save(f2, newf)
