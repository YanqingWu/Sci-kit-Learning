from sklearn import datasets
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
# %matplotlib

def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)



#多元线性回归
def test_LinearRegression(*data):
    X_train,X_test,y_train,y_test = data
    X = np.vstack((X_train,X_test))
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %.2f'  %(regr.coef_, regr.intercept_))
    print('Residual sum of squares : %.2f'  %(np.mean(regr.predict(X_test) - y_test**2)))
    print('Score : %.2f' %regr.score(X_test,y_test))
    predict = regr.predict(X)
    return predict
X_train,X_test,y_train,y_test = load_data()
predict = test_LinearRegression(X_train,X_test,y_train,y_test)

Y = np.vstack((y_train.reshape(y_train.size, 1),y_test.reshape(y_test.size ,1))).ravel()
plt.scatter(Y,predict)
plt.plot(Y,Y)



#使用默认的alpha值做岭回归
def test_Ridge(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %.2f'  %(regr.coef_, regr.intercept_))
    print('Residual sum of squares : %.2f'  %(np.mean(regr.predict(X_test) - y_test**2)))
    print('Score : %.2f' %regr.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_Ridge(X_train,X_test,y_train,y_test)
    


#使用不同的alpha值做岭回归
def test_Ridge_alpha(*data):
    X_train,X_test,y_train,y_test = data
    alphas = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha = alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'score')
    ax.set_xscale('log')
    ax.set_title('Ridge')
    plt.show()
    
X_train,X_test,y_train,y_test = load_data()
test_Ridge_alpha(X_train,X_test,y_train,y_test)



#使用默认的alpha值做lasso回归    
def test_Lasso(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.Lasso()
    regr.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %.2f'  %(regr.coef_, regr.intercept_))
    print('Residual sum of squares : %.2f'  %(np.mean(regr.predict(X_test) - y_test**2)))
    print('Score : %.2f' %regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data()
test_Lasso(X_train,X_test,y_train,y_test)
    
    
    
#使用不同的alpha值做lasso回归
def test_Lasso_alpha(*data):
    X_train,X_test,y_train,y_test = data
    alphas = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha = alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'score')
    ax.set_xscale('log')
    ax.set_title('Lasso')
    plt.show()
    
X_train,X_test,y_train,y_test = load_data()
test_Lasso_alpha(X_train,X_test,y_train,y_test)



#ElasticNet回归的惩罚项是介于L1范数和L2范数之间的一种权衡
#使用默认的alpha，l1_ratio做ElasticNet回归
def test_ElasticNet(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.ElasticNet()
    regr.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %.2f'  %(regr.coef_, regr.intercept_))
    print('Residual sum of squares : %.2f'  %(np.mean(regr.predict(X_test) - y_test**2)))
    print('Score : %.2f' %regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data()
test_ElasticNet(X_train,X_test,y_train,y_test)



#使用不同的alpha，l1_ratio做ElasticNet回归
def test_ElasticNet_alpha_rho(*data):
    X_train,X_test,y_train,y_test = data
    alphas = np.logspace(-2,2)
    rhos = np.linspace(0.01,1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
            regr.fit(X_train,y_train)
            scores.append(regr.score(X_test,y_test))
    alphas,rhos = np.meshgrid(alphas,rhos)
    scores = np.array(scores).reshape(alphas.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas,rhos,scores,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'rho')
    ax.set_zlabel('score')
    ax.set_title('ElasticNet')

X_train,X_test,y_train,y_test = load_data()
test_ElasticNet_alpha_rho(X_train,X_test,y_train,y_test)



#logistic回归，默认one-vs-rest策略
def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train,test_size = 0.25, random_state = 0, 
                                           stratify = y_train)

def test_LogisticRegression(*data):
    X_train,X_test,y_train,y_test = data
    X = np.vstack((X_train,X_test))
    regr = linear_model.LogisticRegression()
    regr.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %s'  %(regr.coef_, regr.intercept_))  
    #此处注意对于多元logistic回归有多个intercept，不能用%0.2f
    print('Score : %.2f' %regr.score(X_test,y_test))
    predict = regr.predict(X)
    return predict
X_train,X_test,y_train,y_test = load_data()
predict = test_LogisticRegression(X_train,X_test,y_train,y_test)




#logistic回归，直接采用多分类逻辑
def test_LogisticRegression_multinomial(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %s'  %(regr.coef_, regr.intercept_))  
    #此处注意对于多元logistic回归有多个intercept，不能用%0.2f
    print('Score : %.2f' %regr.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_LogisticRegression_multinomial(X_train,X_test,y_train,y_test)



#logistic回归，参数C对模型预测性能的影响
def test_LogisticRegression_C(*data):
    X_train,X_test,y_train,y_test = data
    Cs = np.logspace(-2,4,num = 100)
    scores = []
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,scores)
    ax.set_xlabel(r'C')
    ax.set_ylabel(r'score')
    ax.set_xscale('log')
    ax.set_title('LogisticRegression')
    plt.show()
    
X_train,X_test,y_train,y_test = load_data()
test_LogisticRegression_C(X_train,X_test,y_train,y_test)



#线性判别分析，LDA
def test_LinearDiscriminantAnalysis(*data):
    X_train,X_test,y_train,y_test = data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train,y_train)
    print('Coefficients : %s , intercept : %s'  %(lda.coef_, lda.intercept_))  
    #此处注意对于多元logistic回归有多个intercept，不能用%0.2f
    print('Score : %.2f' %lda.score(X_test,y_test))
    
X_train,X_test,y_train,y_test = load_data()
test_LinearDiscriminantAnalysis(X_train,X_test,y_train,y_test)
#################分类画图##################
def plot_LDA(converted_X, y):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers = 'o*s'
    for target,color,marker in zip([0,1,2],colors,markers):
        pos = (y==target).ravel()
        X = converted_X[pos,:]
        ax.scatter(X[:,0],X[:,1],X[:,2],color = color,marker = marker,label = 'Lable %d' %target)
    ax.legend(loc='best')
    fig.suptitle('Iris After LDA')
    plt.show()
    
X_train,X_test,y_train,y_test = load_data()
X = np.vstack((X_train,X_test))
Y = np.vstack((y_train.reshape(y_train.size, 1),y_test.reshape(y_test.size ,1)))
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X,Y)
converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_
plot_LDA(converted_X, Y)
        

    
#考察不同的solver对模型判别的影响
def test_LinearDiscriminantAnalysis_solver(*data):
    X_train,X_test,y_train,y_test = data
    solvers = ['svd','lsqr','eigen']
    for solver in solvers :
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver = solver)
            lda.fit(X_train,y_train)
            print('Score at solver = %s : %.2f' %(solver,lda.score(X_test,y_test)) )

X_train,X_test,y_train,y_test = load_data()
test_LinearDiscriminantAnalysis_solver(X_train,X_test,y_train,y_test)



#考察抖动shrinkage的影响，引入shrinkage相当于引入了正则化
def test_LinearDiscriminantAnalysis_shrinkage(*data):
    X_train,X_test,y_train,y_test = data
    shrinkages = np.linspace(0,1, num = 20)
    scores = []
    for shrinkage in shrinkages:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver = 'lsqr',shrinkage = shrinkage)
        lda.fit(X_train,y_train)
        scores.append(lda.score(X_train,y_train))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(shrinkages,scores)
    ax.set_xlabel(r'shrinkage')
    ax.set_ylabel(r'score')
    ax.set_ylim(0,1.05)
    ax.set_title('LinearDiscriminantAnalysis')
    plt.show()
    
X_train,X_test,y_train,y_test = load_data()
test_LinearDiscriminantAnalysis_shrinkage(X_train,X_test,y_train,y_test)












