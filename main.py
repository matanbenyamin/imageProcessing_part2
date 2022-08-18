import numpy
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
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

# this should be in a try catch block
from kmeans_gpu import KMeans as KM_GPU
import torch

sift = cv2.SIFT_create()



def extract_sift(im_path):
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clean image
    # img = cv2.medianBlur(img, 3)


    kp = sift.detect(img,None)

     # filter by size
    # kp = [k for k in kp if k.size > 1]

    # taje the strongest keypoints
    # kp = sorted(kp, key=lambda k: k.response, reverse=True)[:100]


    kp, des = sift.compute(img, kp)

    return des

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def get_words(X_raw, kmeans, idf):
    vocab_size = kmeans.n_clusters
    vocab = np.array(X_raw[0])
    for x in X_raw:
        t = np.array(x)
        vocab_test = np.concatenate([vocab, t])
    # predict the cluster for each descri[tor in X_raw
    X = np.zeros((len(X_raw), 100))
    for i, x in tqdm.tqdm(enumerate(X_raw)):
        if x is not None:
            if len(x) > 0:
                hist = np.histogram(kmeans.predict(x), bins=100)[0]
                # hist = hist / np.sum(hist)
                X[i] = hist
    #calcualte  tf idf
    # tf = np.sum(X, axis=0)
    # idf = np.sum(X > 0, axis=0)
    # idf = np.log(vocab_size / idf)
    # tf_idf = tf * idf
    X = X * idf[None, :]


    return X

def generte_vocabulary(path = 'data/train/', folder_nums = None):


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

def cluster_vocab_hdbscan(X_raw, vocab_size = 150):
        import hdbscan
        vocab = np.array(X_raw[0])
        for x in X_raw:
            t = np.array(x)
            vocab = np.concatenate([vocab, t])

        # now cluster the words and print
        print("Clustering...")
        hdbscan = hdbscan.HDBSCAN(min_cluster_size=10).fit(vocab)
        print("Clustering done.")

        # predict the cluster for each descri[tor in X_raw
        X = np.zeros((len(X_raw), vocab_size))
        for i, x in tqdm.tqdm(enumerate(X_raw)):
            if x is not None:
                if len(x) > 0:
                    hist = np.histogram(hdbscan.labels_, bins=vocab_size)[0]
                    hist = hist / np.sum(hist)
                    X[i] = hist

        return hdbscan, X


def cluster_vocab_gpu(X_raw, vocab_size = 150):

    vocab = np.array(X_raw[0])
    for x in X_raw:
        t = np.array(x)
        vocab = np.concatenate([vocab, t])
    # Config
    batch_size = vocab.shape[0]
    num_cluster = vocab_size

    # Create KMeans Module
    kmeans = KM_GPU(
        n_clusters=num_cluster,
        tolerance=1e-4,
        distance='euclidean',
        sub_sampling=None,
    )

    x = torch.from_numpy(vocab).float()
    batch_x = x.unsqueeze(dim=0)
    center_pts = kmeans(batch_x)
    center_pts = np.array(center_pts)

    # predict the cluster for each descri[tor in X_raw
    X = np.zeros((len(X_raw), vocab_size))
    for i, x in tqdm.tqdm(enumerate(X_raw)):
        if x is not None:
            if len(x) > 0:
                closest_centroid = np.zeros(vocab.shape[0])
                for ii in range(x.shape[0]):
                    dis = np.min(np.sum((x[ii, :] - center_pts) ** 2, axis=1))
                    if dis < 1e3:
                        closest_centroid[ii] = np.argmin(np.sum((x[ii, :] - center_pts) ** 2, axis=1))
                hist = np.histogram(closest_centroid, bins=vocab_size)[0]
                hist = hist / np.sum(hist)
                X[i] = hist


    return kmeans, X

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
                # hist = hist / np.sum(hist)
                X[i] = hist

    # calcualet tf idf
    tf = np.sum(X, axis=0)
    idf = np.sum(X > 0, axis=0)
    idf = np.log(vocab_size / idf)
    X = X * idf[None, :]

    return kmeans, X, idf


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



def eval(kmeans, model, idf,  path='data/test/', folder_nums = None):

    X_raw_test, y_test = generte_vocabulary(path=path, folder_nums=folder_nums)
    X_test = get_words(X_raw_test, kmeans, idf)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    # imports
    from main import *

    ti = time.time()
    folder_nums = [11, 12,13,14]
    #  all folders
    # folder_nums = None

    X_raw, y = generte_vocabulary(path='data/train/', folder_nums=folder_nums)

    vocab_t = time.time()
    print('vocab_t: {}'.format((vocab_t-ti)/60))
    kmeans, X, idf = cluster_vocab(X_raw, vocab_size=100)
    cluster_t = time.time()
    print('cluster_t: {}'.format((cluster_t - ti) / 60))


    # try a simple knn
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X, y)



    eval(kmeans, knn_model, idf, path='data/test/', folder_nums=folder_nums)





