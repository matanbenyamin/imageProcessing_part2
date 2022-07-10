import numpy
import numpy as np
import cv2
import os

import sklearn as sk
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from collections import Counter
import tqdm
import logging


def generte_vocabulary(path = 'data/data/train/', vocab_size = 150):

    sift = cv2.SIFT_create()
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    # limit classes for development
    subfolders = subfolders[4:8]


    # initialzie vocab
    vocab = np.zeros((0,128))

    X_raw = []
    y = []
    y_raw = []
    count = 0

    for subfolder in tqdm.tqdm(subfolders):
        images = [f.path for f in os.scandir(subfolder) if f.is_file()]
        cl = subfolder.rsplit('/', 1)[-1]
        count  = 0
        for im in images:
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # normalize image
            img = img.astype(float)
            img = img-np.min(img)
            img = img/np.max(img)
            img = img*255
            img = img.astype(np.uint8)

            kp = sift.detect(img,None)
            kp, des = sift.compute(img, kp)

            # filter out weak features
            # des = np.argsort(np.abs(des))[-30:]

            X_raw.append(des)
            y.append(cl)
            count += 1
            if count >5 == 0:
                break

            vocab = np.vstack((vocab, des))
            # add label to each descriptor
            y_raw.append([cl] * len(des))


    # now cluster the words and print
    kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(vocab)

    # predict the cluster for each descri[tor in X_raw
    X = np.zeros((len(X_raw), vocab_size))
    for i, x in enumerate(X_raw):
        hist =  np.bincount(kmeans.predict(x.astype(float)), minlength=vocab_size)
        hist = hist / np.sum(hist)
        X[i] = hist


    return kmeans, X, y

# for test image, get sift features, and assign to nearest cluster
def image_to_histogram(img, kmeans):

    img = cv2.resize(img, (300, 300))

    kp = sift.detect(img,None)
    kp, des = sift.compute(img, kp)
    # filter out weak features
    # des = np.argsort(np.abs(des))[-100:]

    hist = np.bincount(kmeans.predict(des.astype(float)), minlength=kmeans.n_clusters)
    hist = hist / np.sum(hist)
    return hist


def histogram_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)



def classify_image(img, X, y, K = 6):
    # this function classfies img based on data X with knn
    # K is the number of nearest neighbors to use
    # img is the image to classify
    hist = image_to_histogram(img, kmeans)


    distances = np.zeros(len(X))
    for i, x in enumerate(X):
        distances[i] = histogram_distance(hist, x)
    nearest_neighbors = np.argsort(distances)[:K]
    nearest_neighbors_labels = [y[i] for i in nearest_neighbors]
    # print(nearest_neighbors_labels, distances[nearest_neighbors])
    # print(nearest_neighbors_labels)

    nearest_neighbors_labels = Counter(nearest_neighbors_labels).most_common(1)[0][0]
    return nearest_n



def evaluate_classifier(path, sample_size = 100):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    # limit classes for development
    subfolders = subfolders[4:8]

    # randomly select image from each folder
    count = 0
    correct = 0
    for subfolder in subfolders:
        images = [f.path for f in os.scandir(subfolder) if f.is_file()]
        cl = subfolder.rsplit('/', 1)[-1]
        inds = np.random.choice(len(images), sample_size, replace=False)
        for ind in inds:
            img = cv2.imread(images[ind])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(cl, classify_image(img, X, y))
            if classify_image(img, X, y) == cl:
                correct += 1
    return correct / (sample_size*len(subfolders))




# ======================= train
kmeans, X, y = generte_vocabulary(vocab_size=2)

# ======================= Evaluate classifier
print(evaluate_classifier('data/data/test/', sample_size = 5))

# ======================= Optimize vocabulary size
for i in range(50, 250, 100):
    kmeans, X, y = generte_vocabulary(vocab_size=i)
    res = evaluate_classifier('data/data/test/', sample_size = 15)
    print(i, res)


# ================= test with random image
path = 'data/data/test/'

subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
# limit classes for development
subfolders = subfolders[4:8]

# randomly select image from each folder
for subfolder in subfolders:
    images = [f.path for f in os.scandir(subfolder) if f.is_file()]
    cl = subfolder.rsplit('/', 1)[-1]
    img = cv2.imread(images[np.random.randint(0, len(images))])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(cl, classify_image(img, X, y))




# optimize kmeans
for i in range(5, 15, 50):
    kmeans, X, y = generte_vocabulary(vocab_size=i)
    res = evaluate_classifier('data/data/test/', sample_size = 65)
    print(i, res)

# visulazie clusters woth umap
import umap
# random sample from vocabulary
inds = np.random.choice(len(vocab), 2500, replace=False)
X_umap = umap.UMAP(n_components=3, min_dist=0.1 ).fit_transform(vocab[inds])

# fig = px.scatter(X_umap, x=0, y=1)
# fig.show()
import pandas as pd
df = pd.DataFrame(X_umap)
label_dict = {v : k for k, v in enumerate(y)}

fig = px.scatter_3d(df, x=0, y=1, z=2)
fig.show()

# choose best kmeans k woith elbow
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
for k in range(5, 15, 5):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(vocab)
    print(k, silhouette_score(vocab, kmeans.labels_))