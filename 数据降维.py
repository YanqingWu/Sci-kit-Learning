#PCA（主成分分析）
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold
%matplotlib

def load_data():
    iris = datasets.load_iris()
    return iris.data,iris.target,iris.target_names

def test_PCA(*data):
    X,y = data
    pca = decomposition.PCA()
    pca.fit(X)
    print('explained varians ratio : %s' %str(pca.explained_variance_ratio_))

X,y,target_names = load_data()
test_PCA(X,y)

def plot_PCA(*data):
    X,y,target_names = data
    pca = decomposition.PCA(n_components = 2)
    pca.fit(X)
    print('explained varians ratio : %s' %str(pca.explained_variance_ratio_))
    X_r = pca.transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ['navy', 'turquoise', 'darkorange']
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,label=target_name)
    ax.set_title('PCA')
    
X,y,target_names = load_data()
plot_PCA(X,y,target_names)






    