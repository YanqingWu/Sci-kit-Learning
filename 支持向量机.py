#支持向量机本质上是非线性方法，在样本较少时，容易抓住数据和特征之间的非线性关系，可以避免神经网络结构选择和局部最小值，可以解决高纬问题

#线性分类SVM
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model,model_selection,svm
%matplotlib
import numpy as np

def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size = 0.25)

def load_data_classfication():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train,y_train,test_size = 0.25,stratify = y_train)

def test_LinearSVC(*data):
    X_train,X_test,y_train,y_test = data
    cls = svm.LinearSVC()
    cls.fit(X_train,y_train)
    print('Coefficients : %s, intercept : %s' %(cls.coef_,cls.intercept_))
    print('Score : %.2f' % cls.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_classfication()
test_LinearSVC(X_train,X_test,y_train,y_test)

#考察损失函数的影响
def test_LinearSVC_loss(*data):
    X_train,X_test,y_train,y_test = data
    losses = ['hinge','squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss = loss)
        cls.fit(X_train,y_train)
        print('loss = %s' %loss)
        print('Coefficients : %s, intercept : %s' %(cls.coef_,cls.intercept_))
        print('Score : %.2f' % cls.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_classfication()
test_LinearSVC_loss(X_train,X_test,y_train,y_test)

#考察惩罚性的影响
def test_LinearSVC_L12(*data):
    X_train,X_test,y_train,y_test = data
    L12 = ['l1','l2']
    for p in L12:
        cls = svm.LinearSVC(penalty=p,dual = False)
        cls.fit(X_train,y_train)
        print('penalty = %s' %p)
        print('Coefficients : %s, intercept : %s' %(cls.coef_,cls.intercept_))
        print('Score : %.2f' % cls.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_classfication()
test_LinearSVC_L12(X_train,X_test,y_train,y_test)

#考察惩罚系数C的影响
def test_LinearSVC_C(*data):
    X_train,X_test,y_train,y_test = data
    Cs = np.logspace(-2,1)
    training_scores = []
    testing_scores = []
    for C in Cs:
        cls = svm.LinearSVC(C = C)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,training_scores,label = 'training_scores')
    ax.plot(Cs,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classfication()
test_LinearSVC_C(X_train,X_test,y_train,y_test)


#非线性分类SVM

#线性核
def test_SVC_linear(*data):
    X_train,X_test,y_train,y_test = data
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train,y_train)
    print('Coefficients : %s, intercept : %s' %(cls.coef_,cls.intercept_))
    print('Score : %.2f' % cls.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_classfication()
test_SVC_linear(X_train,X_test,y_train,y_test)    
#多项式核，多项式核有三个参数，degree，gamma，coef0    
def test_SVC_poly(*data):
    X_train,X_test,y_train,y_test = data
    fig = plt.figure()
    degrees = range(1,20)
    training_scores = []
    testing_scores = []
    #### 测试 degree ####
    for degree in degrees:
        cls = svm.SVC(kernel='poly',degree = degree)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
        
    ax = fig.add_subplot(1,3,1)
    ax.plot(degrees,training_scores,label = 'training_scores')
    ax.plot(degrees,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.set_ylim(0,1)
    ax.set_title('degree')
    ax.legend()
        
    #### 测试 gamma ####
    gammas = range(1,20)
    training_scores = []
    testing_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel='poly',gamma = gamma,degree=3)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
    ax = fig.add_subplot(1,3,2)
    ax.plot(gammas,training_scores,label = 'training_scores')
    ax.plot(gammas,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.legend()
    ax.set_ylim(0,1)
    ax.set_title('gamma')
        
    #### 测试 coef0 ####
    rfs = range(1,20)
    training_scores = []
    testing_scores = []
    for rf in rfs:
        cls = svm.SVC(kernel='poly',coef0=rf,degree=3,gamma=3)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
    ax = fig.add_subplot(1,3,3)
    ax.plot(rfs,training_scores,label = 'training_scores')
    ax.plot(rfs,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('coef0')
    ax.set_ylim(0,1)

X_train,X_test,y_train,y_test = load_data_classfication()
test_SVC_poly(X_train,X_test,y_train,y_test)


#考察高斯核
def test_SVC_rbf(*data):
    X_train,X_test,y_train,y_test = data
    gammas = range(1,20)
    training_scores = []
    testing_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel='rbf',gamma = gamma)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(gammas,training_scores,label = 'training_scores')
    ax.plot(gammas,testing_scores,label = 'testing_scores')
    ax.legend()
    ax.set_ylim(0,1)
    ax.set_title('gamma')

X_train,X_test,y_train,y_test = load_data_classfication()
test_SVC_rbf(X_train,X_test,y_train,y_test)

#考察sigmoid核
def test_SVC_sigmod(*data):
    X_train,X_test,y_train,y_test = data
    fig = plt.figure()
    #### 测试gamma ####
    gammas = np.logspace(-2,1)
    training_scores = []
    testing_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel='rbf',gamma = gamma)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(gammas,training_scores,label = 'training_scores')
    ax.plot(gammas,testing_scores,label = 'testing_scores')
    ax.legend()
    ax.set_ylim(0,1)
    ax.set_title('gamma')
    
    #### 测试cef0 ####
    rfs = np.linspace(0,5)
    training_scores = []
    testing_scores = []
    for rf in rfs:
        cls = svm.SVC(kernel='poly',coef0=rf,gamma = 0.01)
        cls.fit(X_train,y_train)
        testing_scores.append(cls.score(X_test,y_test))
        training_scores.append(cls.score(X_train,y_train))
        
    ax = fig.add_subplot(1,2,2)
    ax.plot(rfs,training_scores,label = 'training_scores')
    ax.plot(rfs,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('coef0')
    ax.set_ylim(0,1)

X_train,X_test,y_train,y_test = load_data_classfication()
test_SVC_sigmod(X_train,X_test,y_train,y_test)




#线性回归SVM
def test_LinearSVR(*data):
    X_train,X_test,y_train,y_test = data
    regr = svm.LinearSVR()
    regr.fit(X_train,y_train)
    print('Coefficients : %s, intercept : %s' %(regr.coef_,cls.intercept_))
    print('Score : %.2f' % regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_regression()
test_LinearSVR(X_train,X_test,y_train,y_test)  

#考察损失函数的影响
def test_LinearSVR_loss(*data):
    X_train,X_test,y_train,y_test = data
    losses = ['epsilon_insensitive','squared_epsilon_insensitive']
    for loss in losses:
        regr = svm.LinearSVR(loss = loss)
        regr.fit(X_train,y_train)
        print('loss = %s' %loss)
        print('Coefficients : %s, intercept : %s' %(regr.coef_,cls.intercept_))
        print('Score : %.2f' % regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_regression()
test_LinearSVR_loss(X_train,X_test,y_train,y_test)  
        
#考察epsilon对预测的影响
def test_LinearSVR_epsilon(*data):
    X_train,X_test,y_train,y_test = data
    epsilons = np.logspace(-2,2)
    training_scores = []
    testing_scores = []
    for epsilon in epsilons:
        regr = svm.LinearSVR(epsilon = epsilon, loss = 'squared_epsilon_insensitive')
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(epsilons,training_scores,label = 'training_scores')
    ax.plot(epsilons,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('svr')

X_train,X_test,y_train,y_test = load_data_regression()
test_LinearSVR_epsilon(X_train,X_test,y_train,y_test)  
    
#考察惩罚系数C的影响
def test_LinearSVR_C(*data):
    X_train,X_test,y_train,y_test = data
    Cs = np.logspace(-2,2)
    training_scores = []
    testing_scores = []
    for C in Cs:
        regr = svm.LinearSVR(C = C, loss = 'squared_epsilon_insensitive',epsilon = 0.1)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,training_scores,label = 'training_scores')
    ax.plot(Cs,testing_scores,label = 'testing_scores')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('svr')

X_train,X_test,y_train,y_test = load_data_regression()
test_LinearSVR_C(X_train,X_test,y_train,y_test)

#非线性回归svr，同非线性分类svm





