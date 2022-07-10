
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

