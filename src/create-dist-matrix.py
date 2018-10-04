import sys
import os
import numpy as np
from scipy import spatial
import multiprocessing
import time

def get_distance (b):
    d = np.zeros(features.shape[0])
    for i,f in enumerate(features):
        d[i] = spatial.distance.cosine(b, f)
    return d


start = time.time()
filepath = "../features/"
files = sorted(os.listdir(filepath))
features=np.zeros([len(files), 100352])
for i,f in enumerate(files):
    features[i] = np.load(filepath + f)
print ("load time = ", time.time() - start)

basenames = []
for f in files:
    b = ".".join(f.split('.')[:-1])
    basenames.append(b)

def main():
    distances = {}

    start = time.time()
    p = multiprocessing.Pool(3)
    Results = p.map(get_distance, features)
    print("total time = ", time.time() - start)

    distances = np.array(Results)
    np.save("../distances/distances.npy", distances)

    ofile = open("../distances/files.txt", "w")
    for f in files:
        temp = f.split('.')
        res = temp[0] + "." + temp[1]
        print (res, file=ofile)


if __name__=="__main__":
    main()
