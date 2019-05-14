#K-邻近算法，KNN，当k=1时称为最近临近算法

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,datasets,model_selection

def load_classfication_data():
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target
    return model_selection.train_test_split(X_train,y_train,test_size = 0.25,stratify = y_train)

def create_regression_data(n):
    X = np.random.rand(n,1)
    y = np.sin(X).ravel()
    y[::5] += (0.5 - np.random.rand(n/5))
    return model_selection.train_test_split(X,y)

def test_KneighborsClassifier(*data):
    X_train,X_test,y_train,y_test = data
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    clf.kneighbors_graph(X_train,n_neighbors,mode)
    print('Training Score : %f'  %clf.score(X_train,y_train))
    print('Test Score : %f'  %clf.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_classfication_data()
test_KneighborsClassifier(X_train,X_test,y_train,y_test)

#t同时考察K值以及投票规则对预测性能的影响
def test_KneighborsClassifier_k_w(*data):
    X_train,X_test,y_train,y_test = data
    ks = np.linspace(1,y_train.size,num = 100,endpoint=False,dtype='int')
    weights = ['uniform','distance']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for weight in weights:
        training_scores = []
        testing_scores = []
        for k in ks:
            clf = neighbors.KNeighborsClassifier(weights = weight,n_neighbors = k,n_jobs=-1)
            clf.fit(X_train, y_train)
            training_scores.append(clf.score(X_train, y_train))
            testing_scores.append(clf.score(X_test,y_test))
        ax.plot(ks,training_scores,label = 'training_scores : weight = %s' %weight)
        ax.plot(ks,testing_scores,label = 'testing_scores : weight = %s' %weight)
        ax.legend()
        ax.set_xlabel('K')
        ax.set_ylabel('score')
        ax.set_title('KneighborsClassifier')
        ax.set_ylim(0,1.05)
        
X_train,X_test,y_train,y_test = load_classfication_data()
test_KneighborsClassifier_k_w(X_train,X_test,y_train,y_test)   

def test_KneighborsClassifier_k_p(*data):
    X_train,X_test,y_train,y_test = data
    ks = np.linspace(1,y_train.size,num = 100,endpoint=False,dtype='int')
    ps = [1,2,10]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for p in ps:
        training_scores = []
        testing_scores = []
        for k in ks:
            clf = neighbors.KNeighborsClassifier(p = p,n_neighbors = k,n_jobs=-1)
            clf.fit(X_train, y_train)
            training_scores.append(clf.score(X_train, y_train))
            testing_scores.append(clf.score(X_test,y_test))
        ax.plot(ks,training_scores,label = 'training_scores : p = %s' %p)
        ax.plot(ks,testing_scores,label = 'testing_scores : p = %s' %p)
        ax.legend()
        ax.set_xlabel('K')
        ax.set_ylabel('score')
        ax.set_title('KneighborsClassifier')
        ax.set_ylim(0,1.05)
        
X_train,X_test,y_train,y_test = load_classfication_data()
test_KneighborsClassifier_k_p(X_train,X_test,y_train,y_test)   

 
#KNN回归同KNN决策相似，KNeighborsRegressor


            









