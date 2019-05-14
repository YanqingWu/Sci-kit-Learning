#半监督学习就是综合应用有类标记的数据和没有类标记的数据，生成合适的分类函数
import numpy as np
import matplotlib.pyplot as plt
%matplotlib
from sklearn import metrics
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

def load_data():
    digits = datasets.load_digits()
    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    np.random.shuffle(indices)
    X = digits.data[indices]
    y = digits.target[indices]
    n_labeled_points = int(len(y)/10)
    unlabeled_indices = np.arange(len(y))[n_labeled_points:]
    return X,y,unlabeled_indices

#标准的迭代式传播算法
def test_LabelPropagation(*data):
    X,y,unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    
    clf = LabelPropagation(max_iter=100, kernel='rbf', gamma=0.1)
    clf.fit(X,y_train)
    true_labels = y[unlabeled_indices]
    print('Accuracy : %.2f' %clf.score(X[unlabeled_indices],true_labels))
    
data = load_data()
test_LabelPropagation(*data)



#考察alpha和gamma对预测性能的影响
def test_LabelPropagation_alpha_gamma(*data):
    X,y,unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    alphas = np.linspace(0.01,1,num=10,endpoint=True)
    gammas = np.logspace(-2,2,num=5)
    for i,alpha in enumerate(alphas):
        scores = []
        for gamma in gammas:
            clf = LabelPropagation(max_iter=100, kernel='rbf', gamma=gamma, alpha=alpha)
            clf.fit(X,y_train)
            true_labels = y[unlabeled_indices]
            scores.append(clf.score(X[unlabeled_indices],true_labels))
            
        ax.plot(gammas,scores,label = 'alpha = %s' %alpha)
        ax.set_xlabel('gamma')
        ax.set_ylabel('score')
        ax.set_xscale('log')
        ax.legend()
        
data = load_data()
test_LabelPropagation_alpha_gamma(*data)      
        
#考察alpha和KNN核n_neighbors参数的影响 
def test_LabelPropagation_alpha_n_neighbors(*data):
    X,y,unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    alphas = np.linspace(0.01,1,num=2,endpoint=True)
    n_neighbors = [1,2,3,4,5,6,7,8,10,20,30,40,50]
    for i,alpha in enumerate(alphas):
        scores = []
        for n_neighbor in n_neighbors:
            clf = LabelPropagation(max_iter=1000, kernel='knn', n_neighbors=n_neighbor, alpha=alpha)
            clf.fit(X,y_train)
            true_labels = y[unlabeled_indices]
            scores.append(clf.score(X[unlabeled_indices],true_labels))
            
        ax.plot(n_neighbors,scores,label = 'alpha = %s' %alpha)
        ax.set_xlabel('n_neighbors')
        ax.set_ylabel('score')
        ax.set_xscale('log')
        ax.legend()
        
data = load_data()
test_LabelPropagation_alpha_n_neighbors(*data)  


#基于normalized graph laplacian and soft clamping算法
def test_LabelSpreading(*data):
    X,y,unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    clf = LabelSpreading(max_iter=1000, kernel='knn',gamma = 0.1)
    clf.fit(X,y_train)
    true_labels = y[unlabeled_indices]
    predicted_labels = clf.transduction_[unlabeled_indices]
    print('Accuracy : %f' %clf.score(X[unlabeled_indices],true_labels))
    print('Accuracy : %f' %metrics.accuracy_score(true_labels,predicted_labels))

data = load_data()
test_LabelSpreading(*data) 

#考察alpha及gamma的影响
def test_LabelSpreading_alpha_gamma(*data):
    X,y,unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    alphas = np.logspace(-2,-1,num = 10)
    gammas = np.logspace(-2,2,num = 10)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    for i,alpha in enumerate(alphas):
        scores = []
        for gamma in gammas:
            clf = LabelSpreading(max_iter=1000, kernel='knn',gamma = gamma, alpha = alpha)
            clf.fit(X,y_train)
            true_labels = y[unlabeled_indices]
            scores.append(clf.score(X[unlabeled_indices],true_labels))
        
        ax.plot(alphas,scores,label='alpha = %f' %alpha)
        ax.set_xscale('log')
        ax.legend()

data = load_data()
test_LabelSpreading_alpha_gamma(*data)       

#考察alpha和n_neighbors对KNN算法的影响
def test_LabelSpreading_alpha_n_neighbors(*data):
    X,y,unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    alphas = np.linspace(0.01,1,num=2,endpoint=True)
    n_neighbors = [1,2,3,4,5,6,7,8,10,20,30,40,50]
    for i,alpha in enumerate(alphas):
        scores = []
        for n_neighbor in n_neighbors:
            clf = LabelPropagation(max_iter=1000, kernel='knn', n_neighbors=n_neighbor, alpha=alpha)
            clf.fit(X,y_train)
            true_labels = y[unlabeled_indices]
            scores.append(clf.score(X[unlabeled_indices],true_labels))
            
        ax.plot(n_neighbors,scores,label = 'alpha = %s' %alpha)
        ax.set_xlabel('n_neighbors')
        ax.set_ylabel('score')
        ax.set_xscale('log')
        ax.legend()
        
data = load_data()
test_LabelSpreading_alpha_n_neighbors(*data)  

