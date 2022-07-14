import numpy
import numpy as np
import cv2
import os

import sklearn as sk
from sklearn.cluster import KMeans

# import pca
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from collections import Counter
import tqdm
import logging

sift = cv2.SIFT_create()



def extract_sift(im_path):
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clean image
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    kp = sift.detect(img,None)

    #  filter by size
    kp = [k for k in kp if k.size > 5]
    kp, des = sift.compute(img, kp)

    return des

def generte_vocabulary(path = 'data/data/train/', folder_nums = None):


    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    # limit classes for development
    if folder_nums is not None:
        subfolders =  [subfolders[i] for i in folder_nums]


    # initialzie vocab
    vocab = np.zeros((0, 128))

    X_raw = []
    y = []


    for subfolder in tqdm.tqdm(subfolders):

        images = [f.path for f in os.scandir(subfolder) if f.is_file()]
        cl = subfolder.rsplit('/', 1)[-1]

        for im in images:

            des = extract_sift(im)

            if des is not None:
                # add label to each descriptor
                X_raw.append(des)
                y.append(cl)

    return X_raw, y


def cluster_vocab(X_raw, vocab_size = 150):


    vocab = np.array(X_raw[0])
    for x in X_raw:
        t = np.array(x)
        vocab = np.concatenate([vocab, t])

    # now cluster the words and print
    print("Clustering...")
    kmeans = KMeans(n_clusters=vocab_size).fit(vocab)
    print("Clustering done.")

    # predict the cluster for each descri[tor in X_raw
    X = np.zeros((len(X_raw), vocab_size))
    for i, x in tqdm.tqdm(enumerate(X_raw)):
        if x is not None:
            if len(x) > 0:
                hist = np.histogram(kmeans.predict(x), bins=vocab_size)[0]
                hist = hist / np.sum(hist)
                X[i] = hist


    return kmeans, X


def get_random_img(path):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    ind = np.random.choice(len(subfolders))
    images = [f.path for f in os.scandir(subfolders[ind]) if f.is_file()]
    ind = np.random.choice(len(images))
    return images[ind]

# for test image, get sift features, and assign to nearest cluster
def image_to_histogram(img, kmeans):


    des = extract_sift(img)

    # # filter out weak features
    # # des = np.argsort(np.abs(des))[-50:]
    hist = np.histogram(kmeans.predict(des.astype(float)), bins=kmeans.n_clusters, range=(0, kmeans.n_clusters))[0]
    hist = hist / np.sum(hist)

    return hist



def histogram_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)



def classify_image(img, X, y, K = 5):
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
    return nearest_neighbors_labels



def evaluate_classifier(path, sample_size = 100, folder_nums = None):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    # limit classes for development
    if folder_nums is not None:
        subfolders =   [subfolders[i] for i in folder_nums]

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


def eval(X,y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # try a simple knn
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # predict
    y_pred = knn_model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
