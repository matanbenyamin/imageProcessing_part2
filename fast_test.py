import random
import time
from main import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# ======================= train

# [13,14] - 80% with knn,. 93 with svm


# take only folder nums that are in class_df

ti = time.time()
folder_nums = [11,12, 13, 14]


X_raw, y = generte_vocabulary(path='data/train/', folder_nums=folder_nums)

vocab_t = time.time()
print('vocab_t: {}'.format(vocab_t))
kmeans, X = cluster_vocab(X_raw, vocab_size=100)
cluster_t = time.time()
print('cluster_t: {}'.format((cluster_t - ti) / 60))
X_raw_test, y_test = generte_vocabulary(path='data/test/', folder_nums=folder_nums)


vocab_test = np.array(X_raw_test[0])
for x in X_raw_test:
    t = np.array(x)
    vocab_test = np.concatenate([vocab_test, t])
# predict the cluster for each descri[tor in X_raw
X_test = np.zeros((len(X_raw_test), 100))
for i, x in tqdm.tqdm(enumerate(X_raw_test)):
    if x is not None:
        if len(x) > 0:
            hist = np.histogram(kmeans.predict(x), bins=100)[0]
            hist = hist / np.sum(hist)
            X_test[i] = hist

# try a simple knn
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)
y_pred = knn_model.predict(X_test)

print(accuracy_score(y_test, y_pred))

from hiclass import LocalClassifierPerNode
from sklearn.ensemble import RandomForestClassifier
# Use random forest classifiers for every node
rf = RandomForestClassifier()
classifier = LocalClassifierPerNode(local_classifier=knn_model)
# Train local classifier per node
classifier.fit(X, y)
# Predict
y_pred = classifier.predict(X_test)
y_pred = [''.join(x) for x in y_pred]
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





# all folders
ti = time.time()




import pandas as pd
folder_nums = [1, 2, 3, 4, 5, 6]
# list classes by folder in train path
path = 'data/train/'
folder_df = pd.DataFrame()
for i, folder in enumerate(os.listdir(path)):
    if i in folder_nums:
        folder_df = folder_df.append(pd.DataFrame([[folder, i]], columns=['class', 'folder_num']))
classes = folder_df['class'].values

# for each class in level1, create another level
def create_class_sublevel(folder_df, level):
    """
    recieves a list of classes and number of levels
    returns a dataframe with the classes and subclasses
    :param
    :return:
    """
    classes = folder_df['class'].values
    df = pd.DataFrame(classes, columns=['class'])
    df['level{}'.format(1)] = 0
    df.iloc[int(len(classes) / 2):, 1] = 1

    for i in range(2, level):
        df['level{}'.format(i)] = 0
        if i == 2:
            prev_classes = [0,1]
        else:
            prev_classes = np.unique(df['level{}'.format(i - 1)])
        max_class = np.max(prev_classes)
        for j in prev_classes:
            subdf = df[df['level{}'.format(i-1)] == j].copy()
            subdf['level{}'.format(i)] = max_class+1
            max_class += 1
            subdf.iloc[int(len(subdf) / 2):,i]  = max_class+1
            max_class += 1
            df.iloc[df['level{}'.format(i-1)] == j, i] = subdf['level{}'.format(i)]
            end2end(train_path='data/train/', test_path='data/test/', class_df=subdf, level=i)

    return df


new_classes = create_class_sublevel(classes, 3)

def end2end(train_path, test_path, class_df, level):


    # take only folder nums that are in class_df

    ti = time.time()
    X_raw, y = generte_vocabulary(path='data/train/', folder_nums=folder_nums)

    new_y = y.copy()
    for i, yi in enumerate(new_y):
        new_y[i] = class_df.loc[class_df['class']==yi]['level{}'.format(level)].values[0]
    y = new_y

    vocab_t = time.time()
    print('vocab_t: {}'.format(vocab_t))
    kmeans, X = cluster_vocab(X_raw, vocab_size=100)
    cluster_t = time.time()
    print('cluster_t: {}'.format((cluster_t - ti) / 60))
    X_raw_test, y_test = generte_vocabulary(path = 'data/test/', folder_nums = folder_nums)

    new_y = y_test.copy()
    for i, yi in enumerate(new_y):
        new_y[i] = class_df.loc[class_df['class']==yi]['level{}'.format(level)].values[0]
    y_test = new_y

    vocab_test = np.array(X_raw_test[0])
    for x in X_raw_test:
        t = np.array(x)
        vocab_test = np.concatenate([vocab_test, t])
    # predict the cluster for each descri[tor in X_raw
    X_test = np.zeros((len(X_raw_test), 100))
    for i, x in tqdm.tqdm(enumerate(X_raw_test)):
        if x is not None:
            if len(x) > 0:
                hist = np.histogram(kmeans.predict(x), bins=100)[0]
                hist = hist / np.sum(hist)
                X_test[i] = hist


     # try a simple knn
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X, y)
    y_pred = knn_model.predict(X_test)

    print(accuracy_score(y_test, y_pred))




# import silhouette
from sklearn.metrics import silhouette_score

# hdbscan, X = cluster_vocab_hdbscan(X_raw, vocab_size=20)

X_raw_test, y_test = generte_vocabulary(path = 'data/test/', folder_nums = folder_nums)



vocab_test = np.array(X_raw_test[0])
for x in X_raw_test:
    t = np.array(x)
    vocab_test = np.concatenate([vocab_test, t])
# predict the cluster for each descri[tor in X_raw
X_test = np.zeros((len(X_raw_test), 100))
for i, x in tqdm.tqdm(enumerate(X_raw_test)):
    if x is not None:
        if len(x) > 0:
            hist = np.histogram(kmeans.predict(x), bins=100)[0]
            hist = hist / np.sum(hist)
            X_test[i] = hist

# X_test = extractFeatures(kmeans, X_raw_test, len(X_raw_test), 100)
# predict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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








# get X_test
vocab_size = 150
X_test = []
y_test = []
test_path = 'data/test/'
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
path = 'data/test/'

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
    res = evaluate_classifier('data/test/', sample_size = 65)
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