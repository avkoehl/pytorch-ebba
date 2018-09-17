import sys
import os
from math import ceil
import numpy as np
from scipy import spatial
from include.distance import get_distances 
import multiprocessing
from joblib import Parallel, delayed
import pickle

def main():
    filepath = "../features/"
    size = 50 #chunk size
    basenames = []
    num_cores = 4
    distances = {}

    files = sorted(os.listdir(filepath))
    for f in files:
        b = ".".join(f.split('.')[:-1])
        basenames.append(b)

    res = Parallel(n_jobs=num_cores)(delayed(get_distances)(filepath, files, b, size) for b in basenames)

    for r in res:
        distances[r[0]] = r[1]

    with open('distances.pickle', 'wb') as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ofile = open("columns.txt", "w")
    columns = ",".join(basenames)
    print (columns, file=ofile)

if __name__=="__main__":
    main()
