

# ======================= Confusion Matrix
from sklearn.metrics import confusion_matrix

# get predictions
path = 'data/data/test/'
# path = 'data/data/train/'

subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
# limit classes for development
# subfolders = subfolders[:3]
# create dictionary from string labels to numbers


predictions = []
true_y = []
for subfolder in subfolders:
    images = [f.path for f in os.scandir(subfolder) if f.is_file()]
    cl = subfolder.rsplit('/', 1)[-1]
    inds = np.random.choice(len(images), 100, replace=False)
    for ind in inds:
        img = cv2.imread(images[ind])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        predictions.append(classify_image(img, X, y))
        true_y.append(cl)


# classification report
from sklearn.metrics import classification_report
print(classification_report(true_y, predictions))

label_dict = {k: v for v, k in enumerate(set(true_y))}
true_y = [label_dict[k] for k in true_y]
predictions = [label_dict[k] for k in predictions]


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
# add labels from dict
ax.set_xticklabels([''] + list(label_dict.keys()))
ax.set_yticklabels([''] + list(label_dict.keys()))
plt.show()

# multiclass roc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# get predictions
path = 'data/data/test/'
# path = 'data/data/train/'

subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
# limit classes for development
# subfolders = subfolders[:3]


# one vs all roc curve
fpr = dict()
tpr = dict()
roc_auc = dict()
predictions = np.array(predictions)
true_y = np.array(true_y)
for i in range(len(set(true_y))):
    fpr[i], tpr[i], _ = roc_curve(true_y == i, predictions == i)
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plot of a ROC curve for a specific class
plt.figure()
for i in range(len(set(true_y))):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(np.array(label_dict.keys())[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

