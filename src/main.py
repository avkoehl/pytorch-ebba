from include.feature_extractor import *
from math import ceil

def get_distances(fname, features):
    names = []
    dists = []
    for k,v in features.items():
        names.append(k)
        dists.append(spatial.distance.cosine(features[fname],v))

    f,d = sort(names, dists)
    for i,dist in enumerate(d):
        print (f[i], round(1 - dist, 3))

    return f,d

def write_to_files(opath, features):
    for k,v in features.items():
        f = v.data
        np.savetxt(opath + k + ".txt", f)

def main():
    model = VGGNet()
    ipath = "../images/"
    opath = "../features/"
    size = 15 #number of images to process at a time

    images = [f for f in os.listdir(ipath) if ".jpg" in f]
    chunks = [images[x:x+size] for x in range(0, len(images), size)]

    # pass in path, list of filenames
    for i,chunk in enumerate(chunks):
        print ("processing chunk: ", i, "out of ", ceil(len(images) / size))
        features = get_features(ipath, chunk, model)
        write_to_files(opath, features)

if __name__=="__main__":
    main()
