import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import pandas as pd
import random

# load random img
path = 'data/train/'
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
subfolder = random.choice(subfolders)
images = [f.path for f in os.scandir(subfolder) if f.is_file()]
im = random.choice(images)
img = cv2.imread(im)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# normalize image
img = img.astype(float)
img = img - np.min(img)
img = img / np.max(img)
img = img * 255
img = img.astype(np.uint8)
kp = sift.detect(img, None)
kp, des = sift.compute(img, kp)


fig.add_trace(px.scatter_3d(kp, x='pts[0]', y='pts[1]', z='pts[2]'))
fig.show()




Xdf = pd.DataFrame(X)
Xdf['y'] = y
# X_street = Xdf[Xdf['y'] == 'Street']



# plot each class in a separate trace
import plotly.graph_objects as go
fig = go.Figure()
for cl in set(y):
    class_df = Xdf[Xdf['y'] == cl]
    class_df = class_df.drop(columns=['y'])
    mean_df = class_df.mean()
    error = np.array(class_df.std())
    fig.add_trace(go.Scatter(x=mean_df.index, y=mean_df.values, error_y=dict(array=error, thickness=1), name=cl))
fig.show()

# all hists of each class
for cl in set(y):
    class_df = Xdf[Xdf['y'] == cl]
    # class_df = class_df.drop(columns=['y'])
    fig = px.line(np.transpose(np.array(class_df.iloc[:, :-1])), title = cl)
    fig.show()




# plot both
fig = px.line(y=hist1)
fig.add_trace(px.line(y=im_features.ravel(),color_discrete_sequence=['violet']).data[0])
fig.show()

# X_street = Xdf
# plot
 # umap
import umap
umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='correlation').fit_transform(vocab)
fig = px.scatter_3d(umap_embedding, x='umap_embedding[0]', y='umap_embedding[1]', z='umap_embedding[2]')
fig.show()

# sample 100 points from vocab
import random
inds = random.sample(range(0, vocab.shape[0]), 1000)
vocab_s = vocab[inds, :]
fig = px.scatter_3d(x = vocab_s[:,0], y = vocab_s[:,1], z = vocab_s[:,2])
fig.show()