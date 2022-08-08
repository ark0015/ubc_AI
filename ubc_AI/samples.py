import glob

import numpy as np

from ubc_AI.prepfold import pfd

SAMPLE_FILES_DIR = "/data/pulse-learning/Erik/"


def load_pfds(dir=SAMPLE_FILES_DIR):
    SAMPLE_FILES = glob.glob(dir + "*.pfd")
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


def quick_load_samples(*args, **kws):
    # if os.access(SAMPLE_FILES_DIR+"samples.npy", os.R_OK):
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


if __name__ == "__main__":
    samples = load_samples(3)
    for i, s in enumerate(samples):
        np.save(SAMPLE_FILES_DIR + "samples_%s" % i, s)
    # print(quick_load_samples(1)[0].shape)
