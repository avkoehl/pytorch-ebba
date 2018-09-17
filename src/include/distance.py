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

def get_distances(filepath, files, seed, size):
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
    return (seed,d)

