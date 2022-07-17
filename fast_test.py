import random

from main import *
# ======================= train
folder_nums = [11,12,13,14]
# all folders
X_raw, y = generte_vocabulary(folder_nums = folder_nums)

# kmeans, X = cluster_vocab(X_raw, vocab_size=20 )

# import silhouette
from sklearn.metrics import silhouette_score

acc = []
si = []
for vs in list(range(150, 400, 40)):
    kmeans, X = cluster_vocab_gpu(X_raw, vocab_size=vs )
    # calculate silhouette score
    si.append(silhouette_score(vocab, kmeans.labels_, metric='euclidean'))
    acc.append(eval(X,y))
    print('vocab size: {} accuracy: {}'.format(vs, eval(X,y)))


vocab = np.array(X_raw[0])
    for x in X_raw:
        t = np.array(x)
        vocab = np.concatenate([vocab, t])

# visualize 3 dimensions from vocab
import random
#randomsly sample 100 poinbts from vocab
num_points = 10000
indices = random.sample(range(vocab.shape[0]), num_points)
vocab_3d = vocab[indices,:3]
fig = px.scatter_3d(vocab_3d, x=0, y=1, z=2)
fig.show()

fig = px.line(np.transpose(X))
fig.show()

eval(X,y)


def tune_kmeans(all_descriptors):
    """
    Tune sklearn kmeans to get optimal cluster size, which is the codebook size
    :param all_descriptors:
    :return:
    """

    k_list = [5, 10, 20, 40, 60]
    sse = []
    for k in k_list:
        start_ts = datetime.datetime.now()
        print('\nRunning kmeans with cluster {}:'.format(k))
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(all_descriptors)
        sse.append(kmeans.inertia_)
        print('cluster {}: sse is {}'.format(k, sse))
        end_ts = datetime.datetime.now()
        print('time of running : {}'.format(end_ts - start_ts))
        np.save('./output/sse.npy', sse)
    plt.plot(k_list, sse)
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.savefig('./output/tune_kmeans.png')

vocab = np.array(X_raw[0])
for x in X_raw:
    t = np.array(x)
    vocab = np.concatenate([vocab, t])
import datetime
tune_kmeans(vocab)


# classification report
print(classification_report(y_test, y_pred))

# cm
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


fig = px.imshow(confusion_matrix(y_test, y_pred))
fig.show()

# =========== try a simple svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

svm_model = SVC(kernel='poly', C=1)
#different kernels: linear, poly, rbf, sigmoid, precomputed
svm_model.fit(X_train, y_train)

# predict
y_pred = svm_model.predict(X_test)
print(accuracy_score(y_test, y_pred))






# get X_test
vocab_size = 150
X_test = []
y_test = []
test_path = 'data/data/test/'
subfolders = [f.path for f in os.scandir(test_path) if f.is_dir()]
# limit classes for development
# subfolders =  [subfolders[i] for i in folder_nums]
for subfolder in subfolders:
    images = [f.path for f in os.scandir(subfolder) if f.is_file()]
    cl = subfolder.rsplit('/', 1)[-1]
    for im in images:
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_test.append(image_to_histogram(img, kmeans))
        y_test.append(cl)
X_test = np.array(X_test)



# ================= test with random image
path = 'data/data/test/'

subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
# limit classes for development
subfolders = subfolders[4:8]
im = random.choice(subfolders)
images = [f.path for f in os.scandir(im) if f.is_file()]

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