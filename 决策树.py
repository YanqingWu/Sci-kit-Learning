#sklearn有两种决策树，回归决策树和分类决策树，均采用优化的CART决策树算法



#回归决策树
import numpy as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
%matplotlib

def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n,1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return model_selection.train_test_split(X,y,test_size=0.25,random_state=1)
    
def test_DecisionTreeRegressor(*data):
    X_train,X_test,y_train,y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train,y_train)
    print('Training Score : %f'  %regr.score(X_train,y_train))
    print('Test Score : %f'  %regr.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    X = np.arange(0,5,0.01)[:,np.newaxis]
    Y = regr.predict(X)
    ax.scatter(X_train,y_train,label = 'train sample',c='g')
    ax.scatter(X_test,y_test,label = 'test sample',c='r')
    ax.plot(X,Y,label = 'predict value',linewidth=2,alpha=0.5)
    ax.set_xlabel('data')
    ax.set_ylabel('target')
    ax.set_title('Decision Tree Regressor')
    ax.legend(framealpha = 0.5)
    
X_train,X_test,y_train,y_test = creat_data(100)
test_DecisionTreeRegressor(X_train,X_test,y_train,y_test)

#随机划分和最优划分
def test_DecisionTreeRegressor_splitter(*data):
    X_train,X_test,y_train,y_test = data
    splitters = ['best','random']
    for splitter in splitters:
        regr = DecisionTreeRegressor()
        regr.fit(X_train,y_train)
        print('splitter = %s'%splitter)
        print('Training Score : %f'  %regr.score(X_train,y_train))
        print('Test Score : %f'  %regr.score(X_test,y_test))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        X = np.arange(0,5,0.01)[:,np.newaxis]
        Y = regr.predict(X)
        ax.scatter(X_train,y_train,label = 'train sample',c='g')
        ax.scatter(X_test,y_test,label = 'test sample',c='r')
        ax.plot(X,Y,label = 'predict value',linewidth=2,alpha=0.5)
        ax.set_xlabel('data')
        ax.set_ylabel('target')
        title = 'Decision Tree Regressor : spliter = %s' %splitter
        ax.set_title(title)
        ax.legend(framealpha = 0.5)

X_train,X_test,y_train,y_test = creat_data(100)
test_DecisionTreeRegressor_splitter(X_train,X_test,y_train,y_test)

#决策树深度的影响
def test_DecisionTreeRegressor_depth(*data,maxdepth):
    X_train,X_test,y_train,y_test = data
    depths = np.arange(1,maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train,y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label = 'training score')
    ax.plot(depths,testing_scores,label = 'testing score')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    title = 'Decision Tree Regressor'
    ax.set_title(title)
    ax.legend(framealpha = 0.5)

X_train,X_test,y_train,y_test = creat_data(100)
test_DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test,maxdepth=20)






#分类决策树
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train,test_size = 0.25, random_state = 0, 
                                           stratify = y_train)
def test_DecisionTreeClassifier(*data):
    X_train,X_test,y_train,y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print('Training Score : %f'  %clf.score(X_train,y_train))
    print('Test Score : %f'  %clf.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_DecisionTreeClassifier(X_train,X_test,y_train,y_test)

#考察切分质量评价准则对与错的影响
def test_DecisionTreeClassifier_criterion(*data):
    X_train,X_test,y_train,y_test = data
    criterions = ['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        print('Criterion : %s' %criterion)
        print('Training Score : %f'  %clf.score(X_train,y_train))
        print('Test Score : %f'  %clf.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_DecisionTreeClassifier_criterion(X_train,X_test,y_train,y_test)
    
#考察决策树深度的影响
def test_DecisionTreeClassifier_depth(*data,maxdepth):
    X_train,X_test,y_train,y_test = data
    depths = np.arange(1,maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeRegressor(max_depth=depth)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label = 'training score')
    ax.plot(depths,testing_scores,label = 'testing score')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    title = 'Decision Tree Regressor'
    ax.set_title(title)
    ax.legend(framealpha = 0.5)
    
X_train,X_test,y_train,y_test = load_data()
test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth=20)

#画出分类好的树形的图
from sklearn.tree import export_graphviz
import graphviz
X_train,X_test,y_train,y_test = load_data()
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
dot_data = export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
    
    
    
    
    
    