import sys
import os
import numpy as np
from scipy import spatial
import multiprocessing
from joblib import Parallel, delayed
import pickle
import time

def get_distance (b):
    base = features[b]
    d = []
    for k,v in features.items():
        dist = spatial.distance.cosine(base, v)
        d.append(dist)
    return b,d

def load_features(filepath, files):
    features = {}

    for f in files:
        name = ".".join(f.split('.')[:-1])
        vec = np.load(filepath + f)
        features[name] = vec
    return features


start = time.time()
filepath = "../features/"
files = sorted(os.listdir(filepath))
features = load_features(filepath, files)
print ("load time = ", time.time() - start)

basenames = []
for f in files:
    b = ".".join(f.split('.')[:-1])
    basenames.append(b)

def main():
    distances = {}

    start = time.time()
    p = multiprocessing.Pool(3)
    Results = p.map(get_distance, basenames)
    print("total time = ", time.time() - start)

    for r in Results:
        distances[r[0]] = zip(basenames, r[1])

    with open('distances.pickle', 'wb') as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()
