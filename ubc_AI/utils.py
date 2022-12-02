import numpy as np
from scipy import mgrid, ndimage

from ubc_AI.prepfold import pfd


def normalize(data):
    """data:input array of 1-3 dimentions
    to be normalized.
    Remember to return the normalized data.
    The input will not be changed.
    """
    if type(data) in [list]:
        result = []
        for a in data:
            result.append(normalize(a))
        return result
    else:
        if data.ndim > 1:
            N_row = data.shape[0]
            # shape = data.shape
            # return np.array(normalize(data.flatten())).reshape(shape)
            return np.array([normalize(data[i, ...]) for i in range(N_row)])
            # img = np.array([normalize(data[i,...]) for i in range(N_row)])
            # return np.where(img>0, img, 0.)
        else:
            # return PPC.normalize(np.array(data), norm='l1', axis=0)
            # shape = data.shape
            # data = data.reshape((-1,1))
            mean = np.median(data)
            # mean = np.mean(data)
            # mean = np.min(data)
            var = np.std(data)
            # var = np.max(data) - np.min(data)
            if var > 0:
                data = (data - mean) / var
            else:
                data = data - mean
            # data = data.reshape(shape)
        return data


def downsample(a, n, align=0):
    """a: input array of 1-3 dimentions
    n: downsample to n bins
    optional:
    align : if non-zero, downsample grid (coords)
            will have a bin at same location as 'align'
            ( typically max(sum profile) )
            useful for plots vs. phase

    """
    if type(a) in [list]:
        result = []
        for b in a:
            result.append(downsample(b))
        return result
    else:
        shape = a.shape
        D = len(shape)
        if D == 1:
            coords = mgrid[0 : 1 - 1.0 / n : 1j * n]
        elif D == 2:
            d1, d2 = shape
            if align:
                # original phase bins
                x2 = mgrid[0 : 1.0 - 1.0 / d2 : 1j * d2]
                # downsampled phase bins
                crd = mgrid[0 : 1 - 1.0 / n : 1j * n]
                crd += x2[align]
                crd = crd % 1
                crd.sort()
                offset = crd[0] * d2
                coords = mgrid[
                    0 : d1 - 1 : 1j * n, offset : d2 - float(d2) / n + offset : 1j * n
                ]
            else:
                coords = mgrid[0 : d1 - 1 : 1j * n, 0 : d2 - 1 : 1j * n]
        elif D == 3:
            d1, d2, d3 = shape
            coords = mgrid[
                0 : d1 - 1 : 1j * n, 0 : d2 - 1 : 1j * n, 0 : d3 - 1 : 1j * n
            ]
        else:
            raise "too many dimentions %s " % D

        def map_to_index(x, bounds, N):
            xmin, xmax = bounds
            return (x - xmin) / (xmax - xmin) * N

        if D == 1:
            m = len(a)
            x = mgrid[0 : 1 - 1.0 / m : 1j * m]
            if align:
                # ensure new grid lands on max(a)
                coords += x[align]
                coords = coords % 1
                coords.sort()
            # newf = interp(x, a, bounds_error=True)
            # return newf(coords)
            return np.interp(coords, x, a)
        elif D == 2:
            # k,l = a.shape
            # x = mgrid[0:1:1j*k]
            # y = mgrid[0:1:1j*l]
            # f = interp2d(x, y, a)
            # coords = mgrid[0:1:1j*n]
            # return f(coords, coords)
            newf = ndimage.map_coordinates(a, coords, cval=np.median(a))
            return newf
        else:
            coeffs = ndimage.spline_filter(a)
            newf = ndimage.map_coordinates(coeffs, coords, prefilter=False)
            # newf = ndimage.map_coordinates(coeffs, coords )
            return newf


def load_pfds(loaddir=None):
    if not loaddir:
        loaddir = "/data/pulse-learning/Erik/"
    SAMPLE_FILES = glob.glob(loaddir + "*.pfd")
    pfds = []
    for f in SAMPLE_FILES:
        pf = pfd(f)
        pfds.append(pf)
    return pfds


def extractdata(pfds, d, normalize=False, downsample=0):
    """d in [1,2,3]"""
    if d not in [1, 2, 3]:
        raise "d must be in [1,2,3], but assigned %s" % d
    data = []
    for pf in pfds:
        pf.dedisperse()
        profile = pf.profs
        D = len(profile.shape)
        i = D - d
        if i == 1:
            profile = profile.sum(0)
        elif i == 2:
            profile = profile.sum(0).sum(0)
        data.append(profile)
    # data = np.ndarray(data)

    return data


def load_samples(*args, **kws):
    return extractdata(load_pfds(), *args, **kws)


def quick_load_samples(loaddir=None):
    # if os.access(SAMPLE_FILES_DIR+"samples.npy", os.R_OK):
    if not loaddir:
        loaddir = "/data/pulse-learning/Erik/"
    samples = []
    for sf in glob.glob(SAMPLE_FILES_DIR + "samples_*.npy"):
        profile = np.load(sf)
        D = len(profile.shape)
        # print(type(samples), samples.shape)
        if len(args) > 0 and args[0] < 3:
            i = D - args[0]
            if i == 1:
                profile = profile.sum(0)
            elif i == 2:
                # profile = profile.sum(1).T.sum(1)
                profile = profile.sum(0).sum(0)
        samples.append(profile)
    return samples

    # else:
    # return extractdata(load_pfds(), *args, **kws)
