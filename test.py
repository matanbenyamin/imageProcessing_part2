import random
import time
from main import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# ======================= train

# [13,14] - 80% with knn,. 93 with svm


# take only folder nums that are in class_df

ti = time.time()
folder_nums = [11,12]


X_raw, y = generte_vocabulary(path='data/train/', folder_nums=folder_nums)

vocab_t = time.time()
print('vocab_t: {}'.format(vocab_t))
kmeans, X = cluster_vocab(X_raw, vocab_size=100)
cluster_t = time.time()
print('cluster_t: {}'.format((cluster_t - ti) / 60))


X_raw_test, y_test = generte_vocabulary(path='data/test/', folder_nums=folder_nums)
X_test = get_words(X_raw_test, kmeans)


# try a simple knn
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)
y_pred = knn_model.predict(X_test)
print(accuracy_score(y_test, y_pred))


# =========== try a simple svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm_model = SVC(kernel='rbf', C=1)
#different kernels: linear, poly, rbf, sigmoid, precomputed
svm_model.fit(X, y)
# predict
y_pred = svm_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
