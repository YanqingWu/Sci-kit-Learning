#集成学习是机器学习算法中非常强大的工具，有人把它称为机器学习的屠龙刀，非常万能且高效，集成学习融合多种算法，它的思想是‘三个臭皮匠顶过一个诸葛亮’

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,model_selection,ensemble
%matplotlib

def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

def load_data_classifier():
    digites = datasets.load_digits()
    return model_selection.train_test_split(digites.data,digites.target,test_size=0.25,random_state=0)

def test_AdaBoostClassifier(*data):
    X_train,X_test,y_train,y_test = data
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train,y_train)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    estimators_num = len(clf.estimators_)
    X = range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label='Training Score')
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label='Testing Score')
    ax.set_xlabel('estimators_num')
    ax.set_ylabel('score')
    ax.legend()
    ax.set_title('AdaClassifier')

X_train,X_test,y_train,y_test = load_data_classifier()
test_AdaBoostClassifier(X_train,X_test,y_train,y_test)

#考察学习率的影响
def test_AdaBoostClassifier_learning_rate(*data):
    X_train,X_test,y_train,y_test = data
    learning_rates = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    training_scores = []
    testing_scores = []
    for learning_rate in learning_rates:
        clf = ensemble.AdaBoostClassifier(learning_rate=learning_rate,n_estimators=500)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(learning_rates,training_scores,label='Train Score')
    ax.plot(learning_rates,testing_scores,label='Test Score')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('score')
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_AdaBoostClassifier_learning_rate(X_train,X_test,y_train,y_test)
    
#考察algorithm的影响
def test_AdaBoostclassifier_algorithm(*data):
    X_train,X_test,y_train,y_test = data
    algorithms = ['SAMME','SAMME.R']
    fig = plt.figure()
    learning_rates = [0.05,0.1,0.5,0.9]
    for i,learning_rate in enumerate(learning_rates):

        ax = fig.add_subplot(2,2,i+1)
        for algorithm in algorithms:
            clf = ensemble.AdaBoostClassifier(learning_rate=learning_rate,n_estimators=50,algorithm=algorithm)
            clf.fit(X_train,y_train)
            estimators_num = len(clf.estimators_)
            X = range(1,estimators_num+1)
            ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label='algorithm : %s - Training Score'%algorithm)
            ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label='algorithm : %s -Testing Score'%algorithm)

        ax.set_xlabel('estimators_num')
        ax.set_ylabel('score')
        ax.set_title('learning_rate : %f'%learning_rate)
        ax.legend()
    
X_train,X_test,y_train,y_test = load_data_classifier()
test_AdaBoostclassifier_algorithm(X_train,X_test,y_train,y_test)
    
    

    
#Adaboost回归器
def test_AdaboostRegressore(*data):
    X_train,X_test,y_train,y_test = data
    regr = ensemble.AdaBoostRegressor(learning_rate=0.1)
    regr.fit(X_train,y_train)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    estimators_num = len(regr.estimators_)
    X = range(1,estimators_num+1)
    ax.plot(list(X),list(regr.staged_score(X_train,y_train)),label='Training Score')
    ax.plot(list(X),list(regr.staged_score(X_test,y_test)),label='Testing Score')
    ax.set_xlabel('estimators_num')
    ax.set_ylabel('score')
    ax.legend()
    ax.set_title('AdaRegressor')

X_train,X_test,y_train,y_test = load_data_regression()
test_AdaboostRegressore(X_train,X_test,y_train,y_test)    

#考察不同类型分类器的影响
def test_AdaboostRegressor(*data):
    X_train,X_test,y_train,y_test = data
    from sklearn.svm import LinearSVR
    regr_names = ['Tree','SVR']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    regrs = [ensemble.AdaBoostRegressor(),
             ensemble.AdaBoostRegressor(learning_rate=0.1,base_estimator=LinearSVR(epsilon=0.01,C=100))]
    for i,regr in enumerate(regrs):
        regr.fit(X_train,y_train)
        estimators_num = len(regr.estimators_)
        X = range(1,estimators_num+1)
        ax.plot(list(X),list(regr.staged_score(X_train,y_train)),label='regressore = %s - Training Score'%regr_names[i])
        ax.plot(list(X),list(regr.staged_score(X_test,y_test)),label='regressore = %s - Testing Score'%regr_names[i])
        ax.set_xlabel('estimators_num')
        ax.set_ylabel('score')
        ax.set_ylim(0,1)
        ax.legend()
        ax.set_title('AdaRegressor')

X_train,X_test,y_train,y_test = load_data_regression()
test_AdaboostRegressor(X_train,X_test,y_train,y_test)    
        
#考察学习率的影响
def test_Adaboostregressor_learning_rate(*data):
    X_train,X_test,y_train,y_test = data
    learning_rates = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    training_scores = []
    testing_scores = []
    for learning_rate in learning_rates:
        clf = ensemble.AdaBoostRegressor(learning_rate=learning_rate,n_estimators=500)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(learning_rates,training_scores,label='Train Score')
    ax.plot(learning_rates,testing_scores,label='Test Score')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('score')
    ax.legend()   
     
X_train,X_test,y_train,y_test = load_data_regression()
test_Adaboostregressor_learning_rate(X_train,X_test,y_train,y_test)    
    
#考察损失函数的影响
def test_AdaboostRegressor_loss(*data):
    X_train,X_test,y_train,y_test = data
    losses = ['linear','square','exponential']
    learning_rates = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for loss in losses:
        training_scores = []
        testing_scores = []
        for learning_rate in learning_rates:
            regr = ensemble.AdaBoostRegressor(learning_rate=learning_rate,loss=loss)
            regr.fit(X_train,y_train)
            training_scores.append(regr.score(X_train,y_train))
            testing_scores.append(regr.score(X_test,y_test))
        ax.plot(learning_rates,training_scores,label='loss = %s - training_score'%loss)
        ax.plot(learning_rates,testing_scores,label='loss = %s - testing_scores'%loss)
        ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_AdaboostRegressor_loss(X_train,X_test,y_train,y_test)    







#Gradient Tree Boosting,GBDT,梯度提升分类决策树
def test_GradientTreeBoosting(*data):
    X_train,X_test,y_train,y_test = data
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train,y_train)
    print('Training Score : %f'%clf.score(X_train,y_train))
    print('Testing Score : %f'%clf.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_classifier()
test_GradientTreeBoosting(X_train,X_test,y_train,y_test)
    
#考察个体决策树数量的影响
def test_GradientTreeBoosting_num(*data):
    X_train,X_test,y_train,y_test = data
    nums = np.arange(1,100,step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for num in nums:
        clf = ensemble.GradientBoostingClassifier(n_estimators = num)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(nums,training_scores,label='training_score')
    ax.plot(nums,testing_scores,label='testing_scores')
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_GradientTreeBoosting_num(X_train,X_test,y_train,y_test)

#考察决策树深度的影响
def test_GradientTreeBoosting_maxdepths(*data):
    X_train,X_test,y_train,y_test = data
    maxdepths = np.arange(1,20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for maxdepth in maxdepths:
        clf = ensemble.GradientBoostingClassifier(max_depth=maxdepth)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(maxdepths,training_scores,label='training_score')
    ax.plot(maxdepths,testing_scores,label='testing_scores')
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_GradientTreeBoosting_maxdepths(X_train,X_test,y_train,y_test)

#考察学习率的影响
def test_GradientTreeBoosting_learning_rate(*data):
    X_train,X_test,y_train,y_test = data
    learning_rates = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for learning_rate in learning_rates:
        clf = ensemble.GradientBoostingClassifier(learning_rate=learning_rate)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(learning_rates,training_scores,label='training_score')
    ax.plot(learning_rates,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_GradientTreeBoosting_learning_rate(X_train,X_test,y_train,y_test)

#考察subsample的影响，当subsample不等于1时，就是随机梯度提升树，subsample指定提升原始训练集的一个子集用于训练基础决策树，subsample就是子集占原始训练集的大小

def test_GradientTreeBoosting_subsample(*data):
    X_train,X_test,y_train,y_test = data
    subsamples = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for subsample in subsamples:
        clf = ensemble.GradientBoostingClassifier(subsample=subsample)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(subsamples,training_scores,label='training_score')
    ax.plot(subsamples,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()    

X_train,X_test,y_train,y_test = load_data_classifier()
test_GradientTreeBoosting_subsample(X_train,X_test,y_train,y_test)

#考察max_features的影响
def test_GradientTreeBoosting_max_features(*data):
    X_train,X_test,y_train,y_test = data
    max_features = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for max_feature in max_features:
        clf = ensemble.GradientBoostingClassifier(max_feature=max_feature)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(max_features,training_scores,label='training_score')
    ax.plot(max_features,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()    

X_train,X_test,y_train,y_test = load_data_classifier()
test_GradientTreeBoosting_subsample(X_train,X_test,y_train,y_test)


#GradientBoostRegressor,GBRT,梯度提升回归决策树
def test_GradientBoostRegressor(*data):
    X_train,X_test,y_train,y_test = data
    regr = ensemble.GradientBoostingRegressor()
    regr.fit(X_train,y_train)
    print('Training Score : %f'%regr.score(X_train,y_train))
    print('Testing Score : %f'%regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor(X_train,X_test,y_train,y_test)

#考察个体回归树的深度对GBRT的影响
def test_GradientBoostRegressor_n_estimator(*data):
    X_train,X_test,y_train,y_test = data
    nums = np.arange(1,200,step = 2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for n_estimator in nums:
        regr = ensemble.GradientBoostingRegressor(n_estimators=n_estimator)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(nums,training_scores,label='training_score')
    ax.plot(nums,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor_n_estimator(X_train,X_test,y_train,y_test)

#考察回归树的深度的影响
def test_GradientBoostRegressor_maxdepth(*data):
    X_train,X_test,y_train,y_test = data
    maxdepths = np.arange(1,20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for maxdepth in maxdepths:
        regr = ensemble.GradientBoostingRegressor(max_depth=maxdepth)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(maxdepths,training_scores,label='training_score')
    ax.plot(maxdepths,testing_scores,label='testing_scores')
    ax.set_ylim(-1,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor_maxdepth(X_train,X_test,y_train,y_test)    

#考察学习率的影响
def test_GradientBoostRegressor_learning(*data):
    X_train,X_test,y_train,y_test = data
    learning_rates = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for learning_rate in learning_rates:
        regr = ensemble.GradientBoostingRegressor(learning_rate=learning_rate)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(learning_rates,training_scores,label='training_score')
    ax.plot(learning_rates,testing_scores,label='testing_scores')
    ax.set_ylim(-1,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor_learning(X_train,X_test,y_train,y_test) 

#考察subsample的影响
def test_GradientBoostRegressor_subsample(*data):
    X_train,X_test,y_train,y_test = data
    subsamples = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for subsample in subsamples:
        regr = ensemble.GradientBoostingRegressor(subsample=subsample)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(subsamples,training_scores,label='training_score')
    ax.plot(subsamples,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor_subsample(X_train,X_test,y_train,y_test)


#考察损失函数的影响
def test_GradientBoostRegressor_loss(*data):
    X_train,X_test,y_train,y_test = data
    nums = np.arange(1,200,step=2)
    losses = ['ls','lad','huber']
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    alphas = np.linspace(0.01,1,endpoint=False,num=5)
    for alpha in alphas:
        testing_scores = []
        training_scores = []
        for n_estimator in nums:
            regr = ensemble.GradientBoostingRegressor(alpha=alpha,n_estimators=n_estimator,loss='huber')
            regr.fit(X_train,y_train)
            testing_scores.append(regr.score(X_test,y_test))
            training_scores.append(regr.score(X_train,y_train))
        ax.plot(nums,training_scores,label='alpha = %f - training_score'%alpha)
        ax.plot(nums,testing_scores,label='alpha = %f - testing_scores'%alpha)
        ax.set_ylim(0,1.05)
        ax.legend()
    ax = fig.add_subplot(2,1,2)
    for loss in losses[1:]:
        testing_scores = []
        training_scores = []
        for n_estimator in nums:
            regr = ensemble.GradientBoostingRegressor(n_estimators=n_estimator,loss=loss)
            regr.fit(X_train,y_train)
            testing_scores.append(regr.score(X_test,y_test))
            training_scores.append(regr.score(X_train,y_train))
        ax.plot(nums,training_scores,label='loss = %s - training_score'%loss)
        ax.plot(nums,testing_scores,label='loss = %s - testing_scores'%loss)
        ax.set_ylim(0,1.05)
        ax.legend()       

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor_loss(X_train,X_test,y_train,y_test)    

#考察max_features的影响
def test_GradientBoostRegressor_maxfeatures(*data):
    X_train,X_test,y_train,y_test = data
    max_features = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for max_feature in max_features:
        regr = ensemble.GradientBoostingRegressor(max_features=max_feature)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(max_features,training_scores,label='training_score')
    ax.plot(max_features,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_GradientBoostRegressor_maxfeatures(X_train,X_test,y_train,y_test)



#随机森林分类器
def test_RandomForestClassifier(*data):
    X_train,X_test,y_train,y_test = data
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train,y_train)
    print('Training Score : %f'%clf.score(X_train,y_train))
    print('Testing Score : %f'%clf.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_classifier()
test_RandomForestClassifier(X_train,X_test,y_train,y_test)

#考察个体决策树的数量的影响
def test_RandomForestClassifier_n_estimator(*data):
    X_train,X_test,y_train,y_test = data
    n_estimators = np.arange(1,100,step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for n_estimator in n_estimators:
        clf = ensemble.RandomForestClassifier(n_estimators=n_estimator)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(n_estimators,training_scores,label='training_score')
    ax.plot(n_estimators,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_RandomForestClassifier_n_estimator(X_train,X_test,y_train,y_test)

#考察max_depth的影响
def test_RandomForestClassifier_maxdepth(*data):
    X_train,X_test,y_train,y_test = data
    max_depths = np.arange(1,20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for max_depth in max_depths:
        clf = ensemble.RandomForestClassifier(max_depth=max_depth)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(max_depths,training_scores,label='training_score')
    ax.plot(max_depths,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_RandomForestClassifier_maxdepth(X_train,X_test,y_train,y_test)

#考察max_feature的影响
def test_RandomForestClassifier_maxfeatures(*data):
    X_train,X_test,y_train,y_test = data
    max_features = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for max_feature in max_features:
        clf = ensemble.RandomForestClassifier(max_features=max_feature)
        clf.fit(X_train,y_train)
        testing_scores.append(clf.score(X_test,y_test))
        training_scores.append(clf.score(X_train,y_train))
    ax.plot(max_features,training_scores,label='training_score')
    ax.plot(max_features,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_classifier()
test_RandomForestClassifier_maxfeatures(X_train,X_test,y_train,y_test)


#随机森林回归器
def test_RandomForestRegressor(*data):
    X_train,X_test,y_train,y_test = data
    regr = ensemble.RandomForestRegressor()
    regr.fit(X_train,y_train)
    print('Training Score : %f'%regr.score(X_train,y_train))
    print('Testing Score : %f'%regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data_regression()
test_RandomForestRegressor(X_train,X_test,y_train,y_test)
    
#考察个体回归树的数量的影响
def test_RandomForestRegressor_n_estimator(*data):
    X_train,X_test,y_train,y_test = data
    n_estimators = np.arange(1,100,step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for n_estimator in n_estimators:
        regr = ensemble.RandomForestRegressor(n_estimators=n_estimator)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(n_estimators,training_scores,label='training_score')
    ax.plot(n_estimators,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_RandomForestRegressor_n_estimator(X_train,X_test,y_train,y_test)
    
#考察maxdepth的影响
def test_RandomForestRegressor_maxdepth(*data):
    X_train,X_test,y_train,y_test = data
    max_depths = np.arange(1,20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for max_depth in max_depths:
        regr = ensemble.RandomForestRegressor(max_depth=max_depth)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(max_depths,training_scores,label='training_score')
    ax.plot(max_depths,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_RandomForestRegressor_maxdepth(X_train,X_test,y_train,y_test)
    
#考察max_feature的影响  
def test_RandomForestRegressor_maxfeatures(*data):
    X_train,X_test,y_train,y_test = data
    max_features = np.linspace(0.01,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    testing_scores = []
    training_scores = []
    for max_feature in max_features:
        regr = ensemble.RandomForestRegressor(max_features=max_feature)
        regr.fit(X_train,y_train)
        testing_scores.append(regr.score(X_test,y_test))
        training_scores.append(regr.score(X_train,y_train))
    ax.plot(max_features,training_scores,label='training_score')
    ax.plot(max_features,testing_scores,label='testing_scores')
    ax.set_ylim(0,1.05)
    ax.legend()

X_train,X_test,y_train,y_test = load_data_regression()
test_RandomForestRegressor_maxfeatures(X_train,X_test,y_train,y_test)