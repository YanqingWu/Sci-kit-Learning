#感知机只能处理线性可分数据集，人工神经网络的学习就是根据训练数据集来调整神经元之间的链接权重，以及每个功能神经元的阈值

#感知机算法
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib

def create_data(n):
    np.random.seed(1)
    x_11 = np.random.randint(0,100,(n,1))
    x_12 = np.random.randint(0,100,(n,1))
    x_13 = 20 + np.random.randint(0,100,(n,1))
    x_21 = np.random.randint(0,100,(n,1))
    x_22 = np.random.randint(0,100,(n,1))
    x_23 = 10 - np.random.randint(0,100,(n,1))
    
    new_x_12 = x_12 * np.sqrt(2)/2 - x_13 * np.sqrt(2)/2
    new_x_13 = x_12 * np.sqrt(2)/2 + x_13 * np.sqrt(2)/2
    new_x_22 = x_22 * np.sqrt(2)/2 - x_23 * np.sqrt(2)/2
    new_x_23 = x_12 * np.sqrt(2)/2 + x_13 * np.sqrt(2)/2
    
    plus_samples = np.hstack([x_11,new_x_12,new_x_13,np.ones((n,1))])
    minus_samples = np.hstack([x_21,new_x_22,new_x_23,-np.ones((n,1))])
    samples = np.vstack([plus_samples,minus_samples])
    np.random.shuffle(samples)
    return samples

#感知机的原始形式
def perception(train_data,eta,w_0,b_0):
    x = train_data[:,:-1]
    y = train_data[:,-1]
    length = train_data.shape[0]
    eta = eta
    w = w_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while(i<length):
            step_num += 1
            x_i = x[i].reshape((x.shape[1],1))
            y_i = y[i]
            if y_i*(np.dot(np.transpose(w),x_i) + b) <= 0 :
                w = w + eta * y_i * x_i
                b = b + eta * y_i
                break
            else:
                i = i + 1
        if (i == length):
            break
    return (w,b,step_num)

def create_hyperplane(x,y,w,b):
    return (-w[0][0] * x - w[1][0] * y - b) / w[2][0]

def plot_perception(samples,n,w,b):
    fig = plt.figure()
    ax = Axes3D(fig)
    Y = samples[:,-1]
    position_p = Y == 1
    position_m = Y == -1
    ax.scatter(samples[position_p,0],samples[position_p,1],samples[position_p,2],marker = '+',label = '+')
    ax.scatter(samples[position_m,0],samples[position_m,1],samples[position_m,2],marker = '^',label = '-')
    ax.legend()
    x = np.linspace(-30,100,100)
    y = np.linspace(-30,100,100)
    x,y = np.meshgrid(x,y)
    z = create_hyperplane(x,y,w,b)
    ax.plot_surface(x,y,z,rstride = 1, cstride = 1, color = 'g', alpha = 0.2)
    ax.legend()

n = 5
data = create_data(n)
eta,w_0,b_0 = 0.1,np.ones((3,1),dtype=float),1
w,b,num = perception(data,eta,w_0,b_0)
plot_perception(data,n,w,b)

#感知机算法的对偶形式
def creat_w(train_data,alpha):
    x = train_data[:,:-1]
    y = train_data[:,-1]
    N = train_data.shape[0]
    w = np.zeros((x.shape[1],1))
    for i in range(0,N):
        w = w + alpha[i][0] * y[i] *(x[i].reshape(x[i].size,1))
        return w

def perception_dual(train_data,eta,alpha_0,b_0):
    x = train_data[:,:-1]
    y = train_data[:,-1]
    length = train_data.shape[0]
    alpha = alpha_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while(i<length):
            step_num += 1
            x_i = x[i].reshape((x.shape[1],1))
            y_i = y[i]
            w = creat_w(train_data,alpha)
            z = y[i] * (np.dot(np.transpose(w),x_i) + b)
            if z <= 0 :
                alpha[i][0] += eta
                b += eta * y[i]
                break
            else:
                i = i + 1
        if (i == length):
            break
    return (alpha,b,step_num)


n = 3
data = create_data(n)
eta,alpha_0,b_0 = 0.1, np.zeros((data.shape[0] * 2, 1)), 0
alpha,b,num = perception_dual(data,eta,alpha_0,b_0)
w = creat_w(data,alpha)
plot_perception(data,n,w,b)
    
#学习率与收敛速度,
def test_eta(data,etas,w_0,alpha_0,b_0):
    nums1 = []
    nums2 = []
    for eta in etas:
        _,_,num_1 = perception(data,eta,w_0,b_0)
        _,_,num_2 = perception_dual(data,eta,alpha_0,b_0)
        nums1.append(num_1)
        nums2.append(num_2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(etas,nums1,label = 'original iteration times',marker = '+')
    ax.plot(etas,nums2,label = 'dual iteration times',marker = '*')
    ax.set_xlabel('eta')
    ax.set_ylabel('iteration times')
    ax.legend()
    ax.set_title('eta for iteration times')

n = 2
data = create_data(n)
eta,w_0,b_0 = 0.1,np.ones((3,1),dtype=float),1
alpha_0 = np.zeros((data.shape[0] * 2, 1))
etas = np.linspace(0.01,1,num = 25)
test_eta(data,etas,w_0,alpha_0,b_0)


#感知机与线性不可分数据，感知机不收敛
def creat_data_nonlinear(n):
    np.random.seed(1)
    x_11 = np.random.randint(0,100,(n,1))
    x_12 = np.random.randint(0,100,(n,1))
    x_13 = 10 + np.random.randint(0,10,(n,1))
    x_21 = np.random.randint(0,100,(n,1))
    x_22 = np.random.randint(0,100,(n,1))
    x_23 = 20 - np.random.randint(0,10,(n,1))
    
    new_x_12 = x_12 * np.sqrt(2)/2 - x_13 * np.sqrt(2)/2
    new_x_13 = x_12 * np.sqrt(2)/2 + x_13 * np.sqrt(2)/2
    new_x_22 = x_22 * np.sqrt(2)/2 - x_23 * np.sqrt(2)/2
    new_x_23 = x_12 * np.sqrt(2)/2 + x_13 * np.sqrt(2)/2
    
    plus_samples = np.hstack([x_11,new_x_12,new_x_13,np.ones((n,1))])
    minus_samples = np.hstack([x_21,new_x_22,new_x_23,-np.ones((n,1))])
    samples = np.vstack([plus_samples,minus_samples])
    np.random.shuffle(samples)
    return samples   




#多层神经网络与线性不可分数据
def creat_data_no_linear_2d(n):
    x_11 = np.random.randint(0,100,(n,1))
    x_12 = 10 + np.random.randint(-5,5,(n,1))
    x_21 = np.random.randint(0,100,(n,1))
    x_22 = 20 + np.random.randint(0,10,(n,1))
    x_31 = np.random.randint(0,100,(int(n/10),1))
    x_32 = 20 + np.random.randint(0,10,(int(n/10),1))
    
    new_x_11 = x_11 * np.sqrt(2)/2 - x_12 * np.sqrt(2)/2
    new_x_12 = x_11 * np.sqrt(2)/2 + x_12 * np.sqrt(2)/2
    new_x_21 = x_21 * np.sqrt(2)/2 - x_22 * np.sqrt(2)/2
    new_x_22 = x_21 * np.sqrt(2)/2 + x_22 * np.sqrt(2)/2
    new_x_31 = x_31 * np.sqrt(2)/2 - x_32 * np.sqrt(2)/2
    new_x_32 = x_31 * np.sqrt(2)/2 + x_32 * np.sqrt(2)/2    
    
    plus_samples = np.hstack([new_x_11,new_x_12,np.ones((n,1))])
    minus_samples = np.hstack([new_x_21,new_x_22,-np.ones((n,1))])
    err_samples = np.hstack([new_x_31,new_x_32,-np.ones((int(n/10),1))])
    samples = np.vstack([plus_samples,minus_samples,err_samples])   
    np.random.shuffle(samples)
    return samples   

def plot_samples_2d(n):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    samples = creat_data_no_linear_2d(n)
    Y = samples[:,-1]
    position_p = Y == 1
    position_m = Y == -1
    ax.scatter(samples[position_p,0],samples[position_p,1],marker = '+',label = '+',color = 'b')
    ax.scatter(samples[position_m,0],samples[position_m,1],marker = '*',label = '-',color = 'y')
    ax.legend()    

def predict_with_MLPClassfier(data):
    train_data = data 
    samples  = train_data
    train_x = train_data[:,:-1]
    train_y = train_data[:,-1]
    clf = MLPClassifier(activation='logistic',max_iter=1000)
    clf.fit(train_x,train_y)
    print(clf.score(train_x,train_y))
    
    #预测输出
    x_min,x_max = train_x[:,0].min() - 1, train_x[:,0].max() + 2
    y_min,y_max = train_x[:,1].min() - 1, train_x[:,1].max() + 2
    plot_step = 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),np.arange(y_min,y_max,plot_step))
    
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    Y = samples[:,-1]
    position_p = Y == 1
    position_m = Y == -1
    ax.contourf(xx,yy,Z,cmap = plt.cm.Paired)
    ax.scatter(samples[position_p,0],samples[position_p,1],marker = '+',label = '+',color = 'b')
    ax.scatter(samples[position_m,0],samples[position_m,1],marker = '*',label = '-',color = 'y')
    ax.legend() 

data = creat_data_no_linear_2d(500)
predict_with_MLPClassfier(data)



#对鸢尾花进行分类
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets

np.random.seed(0)
iris = datasets.load_iris()
x = iris.data[:,0:2]
y = iris.target
data = np.hstack((x,y.reshape(y.size,1)))
np.random.shuffle(data)

X = data[:,0:2]
Y = data[:,-1]
    
train_x = X[:-30] 
train_y = Y[:-30]
test_x = X[-30:]
test_y = Y[-30:]

def plot_samples(ax,x,y):
    n_classes = 3
    plot_colors = 'byr'
    for i,color in zip(range(n_classes),plot_colors):
        idx = np.where(y == i)
        ax.scatter(x[idx,0],x[idx,1] , c = color, label = iris.target_names[i], cmap = plt.cm.Paired)
        
def plot_classifier_predict_meshgrid(ax,clf,x_min,x_max,y_min,y_max):
    plot_step = 0.02
    xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),np.arange(y_min,y_max,plot_step))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx,yy,Z,cmap = plt.cm.Paired)

def mplclassifier_iris():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    classifier = MLPClassifier(activation='logistic',max_iter=1000,hidden_layer_sizes=(30,))
    classifier.fit(train_x,train_y)
    train_score = classifier.score(train_x,train_y)
    test_score  = classifier.score(test_x,test_y)
    
    x_min,x_max = train_x[:,0].min() - 1, train_x[:,0].max() + 2
    y_min,y_max = train_x[:,1].min() - 1, train_x[:,1].max() + 2
    
    plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
    plot_samples(ax,train_x,train_y)
    
    ax.legend()
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title('train score : %.2f ,test score : %.2f' %(train_score,test_score))
    
mplclassifier_iris()
    
#隐含层对模型预测性能的影响
def mplclassifier_iris_hidden_layer_sizes():
    fig = plt.figure()
    hidden_layer_sizes = [(10,),(30,),(100,),(5,5),(10,10),(30,30)]

    for i,hidden_layer_size in enumerate(hidden_layer_sizes):
        fig.subplots_adjust(wspace =0.4, hspace =0.3)
        ax = fig.add_subplot(2,3,i + 1)
        
        classifier = MLPClassifier(activation='logistic',max_iter=1000,hidden_layer_sizes=hidden_layer_size)
        classifier.fit(train_x,train_y)
        train_score = classifier.score(train_x,train_y)
        test_score  = classifier.score(test_x,test_y)
    
        x_min,x_max = train_x[:,0].min() - 1, train_x[:,0].max() + 2
        y_min,y_max = train_x[:,1].min() - 1, train_x[:,1].max() + 2
    
        plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
        plot_samples(ax,train_x,train_y)
        ax.legend()
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title('layer_size : %s,train score : %.2f ,test score : %.2f' %(hidden_layer_size,train_score,test_score),fontsize = 10)

mplclassifier_iris_hidden_layer_sizes()

#考察激活函数对网络分类器的影响
def mplclassifier_iris_activation():
    fig = plt.figure()
    activations = ['logistic','tanh','relu']

    for i,activation in enumerate(activations):
        ax = fig.add_subplot(1,3,i + 1)
        
        classifier = MLPClassifier(activation=activation,max_iter=1000,hidden_layer_sizes=(30,))
        classifier.fit(train_x,train_y)
        train_score = classifier.score(train_x,train_y)
        test_score  = classifier.score(test_x,test_y)
    
        x_min,x_max = train_x[:,0].min() - 1, train_x[:,0].max() + 2
        y_min,y_max = train_x[:,1].min() - 1, train_x[:,1].max() + 2
    
        plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
        plot_samples(ax,train_x,train_y)
        ax.legend()
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title('activation : %s ,train score : %.2f ,test score : %.2f' 
                     %(activation,train_score,test_score),fontsize = 10)

mplclassifier_iris_activation()

#考察算法对模型的影响
def mplclassifier_iris_algorithms():
    fig = plt.figure()
    algorithms = ['lbfgs', 'sgd', 'adam']

    for i,algorithm in enumerate(algorithms):
        ax = fig.add_subplot(1,3,i + 1)
        classifier = MLPClassifier(activation='tanh',solver = algorithm,max_iter=1000,hidden_layer_sizes=(30,))
        classifier.fit(train_x,train_y)
        train_score = classifier.score(train_x,train_y)
        test_score  = classifier.score(test_x,test_y)
    
        x_min,x_max = train_x[:,0].min() - 1, train_x[:,0].max() + 2
        y_min,y_max = train_x[:,1].min() - 1, train_x[:,1].max() + 2
    
        plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
        plot_samples(ax,train_x,train_y)
        ax.legend()
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title('algorithm : %s ,train score : %.2f ,test score : %.2f' 
                     %(algorithm,train_score,test_score),fontsize = 10)    

mplclassifier_iris_algorithms()

#考察学习率的影响
def mplclassifier_iris_etas():
    fig = plt.figure()
    etas = np.logspace(-4,-1,num = 4)

    for i,eta in enumerate(etas):
        ax = fig.add_subplot(1,4,i + 1)
        classifier = MLPClassifier(activation='tanh',learning_rate_init = eta,max_iter=1000,hidden_layer_sizes=(30,))
        classifier.fit(train_x,train_y)
        train_score = classifier.score(train_x,train_y)
        test_score  = classifier.score(test_x,test_y)
    
        x_min,x_max = train_x[:,0].min() - 1, train_x[:,0].max() + 2
        y_min,y_max = train_x[:,1].min() - 1, train_x[:,1].max() + 2
    
        plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
        plot_samples(ax,train_x,train_y)
        ax.legend()
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1]) 
        ax.set_title('eta : %s ,train score : %.2f ,test score : %.2f' 
                     %(eta,train_score,test_score),fontsize = 8)

mplclassifier_iris_etas()



















    