import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import pandas as pd
import random

# load random img
path = 'data/data/train/'
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


kmeans, X, y = generte_vocabulary(vocab_size = 5)


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






# X_street = Xdf
# plot
