from include.feature_extractor import *

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

    # eventually probably want to chunk this up so not all the features are loaded into memory
    # and write n binary files in numpy where n is the number of chunks
    # something like 10 chunks, once 16000/10 features calculated, write to file then repeat
    # the parallel jpbs are spawned at each repetion, don't handle the chunks in parallel
    features = get_features(ipath, model)
    write_to_files(opath, features)

if __name__=="__main__":
    main()
