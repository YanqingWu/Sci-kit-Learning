#在sklearn里有多个不同的朴素贝叶斯分类器，区别在于假设了不同的后验分布

#朴素贝叶斯分类：
from sklearn import datasets,model_selection,naive_bayes
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib

def show_digits():
    digits = datasets.load_digits()
    fig = plt.figure()
    print('vector from image 0 :',digits.data[0])
    for i in range(25):
        ax = fig.add_subplot(5,5,i+1)
        ax.imshow(digits.images[i],interpolation = 'nearest')

def load_data():
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data,digits.target,test_size = 0.25,stratify =digits.target)

#GaussianNB（高斯贝叶斯分类器），假设特征的条件概率服从高斯分布
def test_GaussianNB(*data):
    X_train,X_test,y_train,y_test = data
    cls = naive_bayes.GaussianNB()
    cls.fit(X_train,y_train)
    print('training score : %.2f' %cls.score(X_train,y_train))
    print('testing score : %.2f' %cls.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data()
test_GaussianNB(X_train,X_test,y_train,y_test)

#MultinomialNB（多项式贝叶斯分类器），假设特征的条件概率服从多项式分布
def test_MultinomialNB(*data):
    X_train,X_test,y_train,y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(X_train,y_train)
    print('training score : %.2f' %cls.score(X_train,y_train))
    print('testing score : %.2f' %cls.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_MultinomialNB(X_train,X_test,y_train,y_test)
#不同的alpha对模型的影响
def test_MultinomialNB_alpha(*data):
    X_train,X_test,y_train,y_test = data
    alphas = np.logspace(-2,4,num = 200)
    train_scores = []
    test_scores = []
    for alpha in alphas :
        cls = naive_bayes.MultinomialNB(alpha = alpha)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,train_scores,label = 'Train Score')
    ax.plot(alphas,test_scores,label = 'Test Score')
    ax.set_xlabel('alphas')
    ax.set_xscale('log')
    ax.set_ylabel('score')
    ax.legend()
    ax.set_title('MultinomialNB')

X_train,X_test,y_train,y_test = load_data()
test_MultinomialNB_alpha(X_train,X_test,y_train,y_test)
    
#BernoulliNB（伯努利贝叶斯分类器），假设特征的条件概率服从二项分布
def test_BernoulliNB(*data):
    X_train,X_test,y_train,y_test = data
    cls = naive_bayes.BernoulliNB()
    cls.fit(X_train,y_train)
    print('training score : %.2f' %cls.score(X_train,y_train))
    print('testing score : %.2f' %cls.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_BernoulliNB(X_train,X_test,y_train,y_test)
    
#不同的alpha值对模型的影响
def test_BernoulliNB_alpha(*data):
    X_train,X_test,y_train,y_test = data
    alphas = np.logspace(-2,5,num = 200)
    train_scores = []
    test_scores = []
    for alpha in alphas :
        cls = naive_bayes.BernoulliNB(alpha = alpha)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,train_scores,label = 'Train Score')
    ax.plot(alphas,test_scores,label = 'Test Score')
    ax.set_xlabel('alphas')
    ax.set_xscale('log')
    ax.set_ylabel('score')
    ax.legend()
    ax.set_title('BernoulliNB')

X_train,X_test,y_train,y_test = load_data()
test_BernoulliNB_alpha(X_train,X_test,y_train,y_test)

#考察binarize参数对模型预测性能的影响
def test_BernoulliNB_binarize(*data):
    X_train,X_test,y_train,y_test = data
    min_x = min(np.min(X_train.ravel()),np.min(X_test.ravel())) - 0.1
    max_x = max(np.max(X_train.ravel()),np.max(X_test.ravel())) + 0.1
    binarizes = np.linspace(min_x,max_x,endpoint=True,num=100)
    train_scores = []
    test_scores = []
    for binarize in binarizes :
        cls = naive_bayes.BernoulliNB(binarize = binarize)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(binarizes,train_scores,label = 'Train Score')
    ax.plot(binarizes,test_scores,label = 'Test Score')
    ax.set_xlabel('alphas')
    ax.set_ylabel('score')
    ax.legend()
    ax.set_title('BernoulliNB')

X_train,X_test,y_train,y_test = load_data()
test_BernoulliNB_binarize(X_train,X_test,y_train,y_test)











