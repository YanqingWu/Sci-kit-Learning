#聚类和EM算法（期望极大算法），用于含有隐变量的概率模型参数估计，具有初值依赖性

import numpy as np
import matplotlib.pyplot as plt
%matplotlib
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture

def create_data(centers, num = 100, std = 0.7):
    X,labels_true = make_blobs(n_samples = num, centers = centers , cluster_std = std)
    return X,labels_true

def plot_data(*data):
    X,labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = 'rgbyckm'
    
    for i,label in enumerate(labels):
        position = labels_true == label
        ax.scatter(X[position,0],X[position,1],label='cluster %d'%label,color = colors[i % len(colors)])
        
    ax.legend(loc = 'best', framealpha = 0.5)
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_title('Cluster')
    
X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
plot_data(X,labels_true)        

#K-means聚类
def test_Kmeans(*data):
    X,labels_true = data
    clst = cluster.KMeans()
    clst.fit(X)
    predicted_labels = clst.predict(X)
    print('ARI : %s' %adjusted_rand_score(labels_true,predicted_labels))
    print('Sum center distance %s' %clst.inertia_)
    
X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_Kmeans(X,labels_true)  

def test_Kmeans_nclusters(*data):
    X,labels_true = data
    nums = range(1,50)
    ARIs = []
    Distances = []
    
    for num in nums:
        clst = cluster.KMeans(n_clusters = num)
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Distances.append(clst.inertia_)
        
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs,marker = '+')
    ax.set_xlabel('n_cluster')
    ax.set_ylabel('ARI')
    
    ax = fig.add_subplot(1,2,2)
    ax.plot(nums,Distances,marker = 'o')
    ax.set_xlabel('n_cluster')
    ax.set_ylabel('inertia_')
    fig.suptitle('KMeans')

X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_Kmeans_nclusters(X,labels_true) 
    
def test_Kmeans_n_init(*data):
    X,labels_true = data
    nums = range(1,50)
    inits = ['random','k-means++']
    i = [0,1]
    fig = plt.figure()
    for i in i:
        ARIs = []
        Distances = []
        for num in nums:
            clst = cluster.KMeans(n_init = num, init = inits[i])
            clst.fit(X)
            predicted_labels = clst.predict(X)
            ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances.append(clst.inertia_)
            
        ax = fig.add_subplot(1,2,1)
        ax.plot(nums,ARIs,marker = '+',label = inits[i])
        ax.set_xlabel('n_init')
        ax.set_ylabel('ARI')
        ax.legend()
        
        ax = fig.add_subplot(1,2,2)
        ax.plot(nums,Distances,marker = 'o',label = inits[i])
        ax.set_xlabel('n_init')
        ax.set_ylabel('inertia_')
        ax.legend()
        fig.suptitle('KMeans')


X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_Kmeans_n_init(X,labels_true) 

    
#密度聚类，DBSCAN
def test_DBSCAN_epsilon(*data):
    X,labels_true = data
    epsilons = np.logspace(-1,1.5)
    ARIs = []
    Core_nums = []
    
    for epsilon in epsilons:
        clst = cluster.DBSCAN(eps = epsilon)
        predicted_labels = clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(epsilons,ARIs,marker = '+')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('ARI')
    ax.set_xscale('log')
    
    ax = fig.add_subplot(1,2,2)
    ax.plot(epsilons,Core_nums,marker = 'o')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Core Numbers')
    fig.suptitle('DBSCAN')

X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_DBSCAN_epsilon(X,labels_true) 

#考察MinPts对密度聚类的影响
def test_DBSCAN_MinPts(*data):
    X,labels_true = data
    min_samples = range(1,100)
    ARIs = []
    Core_nums = []
    for num in min_samples:
        clst = cluster.DBSCAN(min_samples = num)
        predicted_labels = clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(min_samples,ARIs,marker = '+')
    ax.set_xlabel('min_samples')
    ax.set_ylim(0,1)
    ax.set_ylabel('ARI')
    
    ax = fig.add_subplot(1,2,2)
    ax.plot(min_samples,Core_nums,marker = 'o')
    ax.set_xlabel('min_samples')
    ax.set_ylabel('Core Numbers')
    fig.suptitle('DBSCAN')    

X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_DBSCAN_MinPts(X,labels_true)                         


#层次聚类
def test_AgglomerativeClustering(*data):
    X,labels_true = data
    clst = cluster.AgglomerativeClustering()
    predicted_labels = clst.fit_predict(X)
    print('ARI : %s' %adjusted_rand_score(labels_true,predicted_labels))
    
centers = [[1,1],[2,2],[1,2],[10,20]]
X,labels_true = create_data(centers,1000,0.5)
test_AgglomerativeClustering(X,labels_true)

#考察簇的数量对模型预测性能的影响
def test_AgglomerativeClustering_nclusters(*data):
    X,labels_true = data
    nums = range(1,50)
    ARIs = []
    for n_clusters in nums :
        clst = cluster.AgglomerativeClustering(n_clusters = n_clusters)
        predicted_labels = clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(nums,ARIs,marker = '+')
    ax.set_xlabel('n_Cluster')
    fig.suptitle('AgglomerativeClustering')

centers = [[1,1],[2,2],[1,2],[10,20]]
X,labels_true = create_data(centers,1000,0.5)
test_AgglomerativeClustering_nclusters(X,labels_true)
    
#考察链接方式对模型预测性能的影响
def test_AgglomerativeClustering_linkage(*data):
    X,labels_true = data
    nums = range(1,50)
    linkages = ['ward','complete','average']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    markers = '+o*'
    for i,linkage in enumerate(linkages):
        ARIs = []
        for num in nums:
            clst = cluster.AgglomerativeClustering(n_clusters = num,linkage = linkage)
            predicted_labels = clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        
        ax.plot(nums,ARIs,marker = markers[i],label ='linkage = %s' %linkage)
        ax.set_xlabel('n_clusters')
        ax.set_ylabel('ARI')
        ax.legend()
        ax.set_title('AgglomerativeClustering')

centers = [[1,1],[2,2],[1,2],[10,20]]
X,labels_true = create_data(centers,1000,0.5)
test_AgglomerativeClustering_linkage(X,labels_true)



#GaussianMixture混合高斯模型，默认的GMM只有一个簇需要注意
def test_GMM(*data):
    X,labels_true = data
    clst = mixture.GaussianMixture()
    clst.fit_predict(X)
    print('ARI : %s' %adjusted_rand_score(labels_true,predicted_labels))

centers = [[1,1],[2,2],[1,2],[10,20]]
X,labels_true = create_data(centers,1000,0.5)
test_GMM(X,labels_true)

#不同簇对模型预测性能的影响
def test_GMM_n_components(*data):
    X,labels_true = data
    ARIs = []
    nums = range(1,50)
    for num in nums:
        clst = mixture.GaussianMixture(n_components = num)
        predicted_labels = clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(nums,ARIs,marker = '+')
    ax.set_xlabel('n_components')
    ax.set_ylabel('ARI')
    ax.set_title('GMM')
    
centers = [[1,1],[2,2],[1,2],[10,20]]
X,labels_true = create_data(centers,1000,0.5)
test_GMM_n_components(X,labels_true)

#考察协方差类型对模型预测性能的影响
def test_GMM_cov_type(*data):
    X,labels_true = data
    covs = ['spherical','tied','diag','full']
    nums = range(1,50)
    fig = plt.figure()
    markers = '+o*s'
    colors = 'rgbyckm'
    ax = fig.add_subplot(1,1,1)
    for i,cov in enumerate(covs):
        AIRs = []
        for num in nums:
            clst = mixture.GaussianMixture(n_components = num, covariance_type = cov)
            predicted_labels = clst.fit_predict(X)
            AIRs.append(adjusted_rand_score(labels_true,predicted_labels))
            
        ax.plot(nums, AIRs, marker = markers[i], color = colors[i],label = 'cov_type = %s' %cov)
        ax.set_xlabel('n_components')
        ax.set_ylabel('ARI')
        ax.legend()
        fig.suptitle('GMM')
        
centers = [[1,1],[2,2],[1,2],[10,20]]
X,labels_true = create_data(centers,1000,0.5)
test_GMM_cov_type(X,labels_true)









