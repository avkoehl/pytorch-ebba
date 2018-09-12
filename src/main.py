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

    images = [f for f in os.listdir(ipath) if ".jpg" in f]
    chunks = [images[x:x+10] for x in range(0, len(images), 10)]

    # pass in path, list of filenames
    for i,chunk in enumerate(chunks):
        features = get_features(ipath, chunk, model)
        write_to_files(opath, features)

if __name__=="__main__":
    main()
