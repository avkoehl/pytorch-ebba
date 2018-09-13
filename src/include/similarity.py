import sys
import os
from math import ceil
import numpy as np
from scipy import spatial

def get_distance (base, fnames, features):
    d = []
    for i, row in enumerate(features):
        dist = spatial.distance.cosine(base, row)
        d.append(dist)

    return d

def sort (fnames, dists):
    f = np.array(fnames)
    d = np.array(dists)
    inds = d.argsort()
    return f[inds], d[inds]

def get_distances(filepath, seed, opath, size):
    ofile = open(opath + seed, "w") 
    files = os.listdir(filepath)
    chunks = [files[x:x+size] for x in range(0, len(files), size)]

    features = []
    d = []
    fnames = []
    base = 0

    for i,chunk in enumerate(chunks):
        for f in chunk:
            name = ".".join(f.split('.')[:-1])
            vec = np.load(filepath + f)
            if name == seed:
                base = vec
            fnames.append(name)
            features.append(vec)

    d = get_distance(base,fnames,features)
    f,d = sort(fnames, d)
    for i,dist in enumerate(d):
        print (f[i], round(1-dist, 3), file=ofile)

