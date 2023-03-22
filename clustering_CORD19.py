import pandas as pd
import time
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
# Dimension reduction and clustering libraries
import umap
import numpy as np

d = pd.read_csv("CORD19_ae_DEC.csv")
#d = pd.read_csv("CORD19_sae.csv")
#d = pd.read_csv("CORD19.csv")

d=np.array(d)
print(np.shape(d))
#d=d[:10000]
k=10
time_start = time.time()
#embedding = PCA(n_components=2).fit_transform(d)
embedding= umap.UMAP(n_neighbors=2, min_dist=0.2).fit_transform(d)
from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(10, 10)})
# colors
palette = sns.hls_palette(9, l=.4, s=.9)
# plot
#sns.scatterplot(embedding[:,0], embedding[:,1], hue=y_pred, legend='full', palette=palette)
plt.scatter(embedding[:, 0], embedding[:, 1]);
plt.show();
kmeans = cluster.KMeans(n_clusters=k)
y_pred_kmeans = kmeans.fit_predict(embedding)
#Agglomerative_labels = cluster.AgglomerativeClustering(n_clusters=k).fit_predict(embedding)
print('UMAP runtime: {} seconds'.format(time.time()-time_start))
#y_pred=np.argmax(y_pred_kmeans, axis=1)

plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred_kmeans, s=1, cmap='Spectral');
#plt.title('JoLMaProC')
plt.savefig("improved_cluster_JoLMaProC.png")
plt.show();

import numpy as np
from scipy.spatial.distance import euclidean, cdist, pdist, squareform

def db_index(X, y):
    """
    Davies-Bouldin index is an internal evaluation method for
    clustering algorithms. Lower values indicate tighter clusters that
    are better separated.
    """
    # get unique labels
    if y.ndim == 2:
        y = np.argmax(axis=1)
    uniqlbls = np.unique(y)
    n = len(uniqlbls)
    # pre-calculate centroid and sigma
    centroid_arr = np.empty((n, X.shape[1]))
    sigma_arr = np.empty((n,1))
    dbi_arr = np.empty((n,n))
    mask_arr = np.invert(np.eye(n, dtype='bool'))
    for i,k in enumerate(uniqlbls):
        Xk = X[np.where(y==k)[0],...]
        Ak = np.mean(Xk, axis=0)
        centroid_arr[i,...] = Ak
        sigma_arr[i,...] = np.mean(cdist(Xk, Ak.reshape(1,-1)))
    # compute pairwise centroid distances, make diagonal elements non-zero
    centroid_pdist_arr = squareform(pdist(centroid_arr)) + np.eye(n)
    # compute pairwise sigma sums
    sigma_psum_arr = squareform(pdist(sigma_arr, lambda u,v: u+v))
    # divide
    dbi_arr = np.divide(sigma_psum_arr, centroid_pdist_arr)
    # get mean of max of off-diagonal elements
    dbi_arr = np.where(mask_arr, dbi_arr, 0)
    dbi = np.mean(np.max(dbi_arr, axis=1))
    return dbi
dbi=db_index(d, y_pred_kmeans)
print("dbi= ", dbi)
# function to print out classification model report
def classification_report(model_name, test, pred):
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='macro')) * 100), "%")
    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='macro')) * 100), "%")
    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='macro')) * 100), "%")

from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test, y_train, y_test = train_test_split(d,y_pred_kmeans, test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier

# SGD instance
sgd_clf = SGDClassifier(max_iter=10000, tol=1e-3, random_state=42, n_jobs=-1)
# train SGD
sgd_clf.fit(X_train, y_train)

# cross validation predictions
sgd_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3, n_jobs=-1)

# print out the classification report
classification_report("Stochastic Gradient Descent Report (Training Set)", y_train, sgd_pred)

# cross validation predictions
sgd_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=3, n_jobs=-1)

# print out the classification report
classification_report("Stochastic Gradient Descent Report (Test Set)", y_test, sgd_pred)

sgd_cv_score = cross_val_score(sgd_clf, d, y_pred_kmeans, cv=10)
print("Mean cv Score - SGD: {:,.3f}".format(float(sgd_cv_score.mean()) * 100), "%")
