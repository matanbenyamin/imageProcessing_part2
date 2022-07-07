import numpy
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from collections import Counter
import tqdm


def generte_vocabulary(path = 'data/data/train/', vocab_size = 50):

    sift = cv2.SIFT_create()
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

    # initialzie vocab
    vocab = np.zeros((0,128))

    X_raw = []
    y = []

    for subfolder in tqdm.tqdm(subfolders):
        count = 0
        images = [f.path for f in os.scandir(subfolder) if f.is_file()]
        cl = subfolder.rsplit('/', 1)[-1]
        for im in images:
            img = cv2.imread(im, 0)
            # img = img[:, :, ::-1]
            kp = sift.detect(img,None)
            kp, des = sift.compute(img, kp)

            X_raw.append(des)
            y.append(cl)

            vocab = np.vstack((vocab, des))
            count += 1
            if count>=100:
                break

    kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(vocab)

    # quantize vocabulary
    X = np.zeros((len(X_raw), vocab_size))
    for i, x in enumerate(X_raw):
        hist = np.zeros(vocab_size)
        for d in x:
            hist[kmeans.predict(d.reshape(1, -1).astype(float))] += 1
        hist = hist / np.sum(hist)
        X[i] = (hist)

    return kmeans, X, y

# for test image, get sift features, and assign to nearest cluster
def image_to_histogram(img, kmeans):
    # img = cv2.imread(image)
    # img = img[:, :, ::-1]
    kp = sift.detect(img,None)
    kp, des = sift.compute(img, kp)
    histogram = np.zeros(vocab_size)
    for d in des:
        histogram[kmeans.predict(d.reshape(1,-1).astype(float))] += 1
    histogram = histogram / np.sum(histogram)

    return histogram


def histogram_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)



def classify_image(img, X, y, K = 3):
    # this function classfies img based on data X with knn
    # K is the number of nearest neighbors to use
    # img is the image to classify
    hist = image_to_histogram(img, kmeans)
    distances = np.zeros(len(X))
    for i, x in enumerate(X):
        distances[i] = histogram_distance(hist, x)
    nearest_neighbors = np.argsort(distances)[:K]
    nearest_neighbors_labels = [y[i] for i in nearest_neighbors]
    print(nearest_neighbors_labels)
    nearest_neighbors_labels = Counter(nearest_neighbors_labels).most_common(1)[0][0]
    return nearest_neighbors_labels


# ======================= train
kmeans, X, y = generte_vocabulary()

# ================= test with random image
path = 'data/data/test/'
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
# randomly select image from each folder
for subfolder in subfolders:
    images = [f.path for f in os.scandir(subfolder) if f.is_file()]
    cl = subfolder.rsplit('/', 1)[-1]
    img = cv2.imread(images[np.random.randint(0, len(images))], 0)
    print(cl, classify_image(img, X, y))



# reqrite the whole thing in oop
class imageClassifier():
    def __init__(self, vocab_size = 50):
        self.kmeans = None
        self.X = None
        self.y = None
        self.sift = cv2.SIFT_create()
        self.vocab_size = vocab_size

    pass

    def train(self, X, y):
        self.kmeans, self.X, self.y = generte_vocabulary(X, y)
        pass

    def image_to_histogram(self, img):
        kp = self.sift.detect(img,None)
        kp, des = self.sift.compute(img, kp)
        histogram = np.zeros(self.vocab_size)
        for d in des:
            histogram[self.kmeans.predict(d.reshape(1,-1).astype(float))] += 1
        # count to frequency
        histogram = histogram / np.sum(histogram)

        return histogram

    def classify_image(self, img):
        hist = image_to_histogram(img, self.kmeans)
        distances = np.array([histogram_distance(hist, x) for x in self.X])
        return self.y[np.argmin(distances)]

