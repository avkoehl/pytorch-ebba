from include.feature_extractor import *
from math import ceil
import multiprocessing
from joblib import Parallel, delayed

def main():
    model = VGGNet()
    ipath = "../images/"
    opath = "../features/"
    size = 10 #number of images to process at a time
    num_cores = 2

    images = [f for f in os.listdir(ipath) if ".jpg" in f]
    chunks = [images[x:x+size] for x in range(0, len(images), size)]

    # pass in path, list of filenames
    for i,chunk in enumerate(chunks):
        print ("processing chunk: ", i, "out of ", ceil(len(images) / size))
        flist = []
        features = {}

        flist = (Parallel(n_jobs=num_cores)(delayed(get_features)(ipath, image, model) for image in chunk))
        for f in flist:
            features[f[0]] = f[1] 

        print ("writing chunk: ", i, "to files")
        write_to_files(opath, features)

if __name__=="__main__":
    main()
