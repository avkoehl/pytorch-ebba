import sys
import os
from math import ceil
import numpy as np
from scipy import spatial
from include.similarity import sort, get_distances 
import multiprocessing
from joblib import Parallel, delayed

def main():
    filepath = "../features/"
    opath = "../similarities/"
    files = os.listdir(filepath)
    size = 50 #chunk size
    basenames = []
    num_cores = 2

    for f in files:
        b = ".".join(f.split('.')[:-1])
        basenames.append(b)
    Parallel(n_jobs=num_cores)(delayed(get_distances)(filepath, b, opath, size) for b in basenames)

if __name__=="__main__":
    main()
