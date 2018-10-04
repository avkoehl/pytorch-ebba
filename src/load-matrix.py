import numpy as np


def zip_sort(files, dists):
    idxs = dists.argsort()
    files = np.array(files)
    f = files[idxs]
    d = dists[idxs]
    return f,d

def distance(alldistances, inds, files, img):
    idx = inds[img]
    dists = alldistances[idx]
    sorted_files,sorted_dists = zip_sort(files,dists)

    results = []
    for i,f in enumerate(sorted_files):
        res = f + " " + str(sorted_dists[i])
        results.append(res)

    return ",".join(results)

def load_files(filepath):
    ifile = open(filepath, "r")
    files = []
    for line in ifile:
        files.append(line.rstrip())
    return files

distances = np.load("../distances/distances.npy")
files = load_files("../distances/files.txt")

inds = {}
for i,f in enumerate(files):
    inds[f] = i

img = "A-20707-30.jpg"

print (distance(distances, inds, files, img))


