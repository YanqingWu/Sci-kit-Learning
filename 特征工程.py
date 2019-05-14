#二元化
from sklearn.preprocessing import Binarizer
X = [
    [1,2,3,4,5],
    [5,4,3,2,1],
    [3,3,3,3,3],
    [1,1,1,1,1]
    ]

print('before transform : \n',X)
binarizer = Binarizer(threshold=2.5)
print('after transform : \n',binarizer.transform(X))



#独热码
from sklearn.preprocessing import OneHotEncoder
X = [
    [1,2,3,4,5],
    [5,4,3,2,1],
    [3,3,3,3,3],
    [1,1,1,1,1]
    ]
print('before transform : \n',X)
encoder = OneHotEncoder(sparse=False)
encoder.fit(X)
print('active_features_ : \n',encoder.active_features_)
print('feature_indices_ : \n',encoder.feature_indices_)
print('n_values_ : \n',encoder.n_values_)
print('after transform : \n',encoder.transform([[1,2,3,4,5]]))

#min-max标准化
from sklearn.preprocessing import MinMaxScaler
X = [
    [1,5,1,2,10],
    [2,6,3,2,7],
    [3,7,5,6,4],
    [4,8,7,8,1]
    ]
print('before transform : \n',X)
scale = MinMaxScaler(feature_range=(0,1))
scale.fit(X)
print('min_ is : \n',scale.min_)
print('scale_ is : \n',scale.scale_)
print('data_max is : \n',scale.data_max_)
print('data_min is : \n',scale.data_min_)
print('data_range_ is : \n',scale.data_range_)
print('after transform : \n',scale.transform(X))

#MaxabsScaler
from sklearn.preprocessing import MaxAbsScaler
X = [
    [1,5,1,2,10],
    [2,6,3,2,7],
    [3,7,5,6,4],
    [4,8,7,8,1]
    ]
scale = MaxAbsScaler()
scale.fit(X)
scale.transform(X)

#z-score标准化
from sklearn.preprocessing import StandardScaler
X = [
    [1,5,1,2,10],
    [2,6,3,2,7],
    [3,7,5,6,4],
    [4,8,7,8,1]
    ]
scale = StandardScaler()
scale.fit(X)
scale.transform(X)


#正则化
from sklearn.preprocessing import Normalizer
X = [
    [1,2,3,4,5],
    [5,4,3,2,1],
    [3,3,3,3,3],
    [1,1,1,1,1]
    ]
nomolizer = Normalizer(norm='l2')
nomolizer.transform(X)

#过滤式特征选取：VarianceThreshold，剔除方差小于给定阈值的特征
from sklearn.feature_selection import VarianceThreshold
X = [
    [1,2,3,4,5],
    [5,4,3,2,1],
    [3,3,3,3,3],
    [1,1,1,1,1]
    ]
selected = VarianceThreshold(2)
selected.fit(X)
selected.transform(X)

#过滤式特征选取：单变量特征提取,SelectBest保留在某统计指标上得分最高的k个指标，SelectPercentile则保留百分比
from sklearn.feature_selection import SelectKBest,f_classif
X = [
    [1,2,3,4,5],
    [5,4,3,2,1],
    [3,3,3,3,3],
    [1,1,1,1,1]
    ]
y = [0,1,0,1]
selector = SelectKBest(score_func = f_classif,k=3)
selector.fit(X,y)
print('scores_ : \n',selector.scores_)
print('pvalues_ : \n',selector.pvalues_)
print('selected index : \n',selector.get_support(True))
print('after transform : \n',selector.transform(X))

#包裹式特征选取：REF
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn import model_selection
iris = load_iris()
X = iris.data
y = iris.target
estimator = LinearSVC()
selector = RFE(estimator = estimator,n_features_to_select= 2)
selector.fit(X,y)
print('N_features %s' %selector.n_features_)
print('support is %s' %selector.support_)
print('ranking %s' %selector.ranking_)

X_transform = selector.fit_transform(X,y)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25,stratify = y)
X_transform_train,X_transform_test,y_train_t,y_test_t = model_selection.train_test_split(X_transform,y,
                                                                                     test_size=0.25,stratify = y)

clf = LinearSVC()
clf_transform = LinearSVC()
clf.fit(X_train,y_train)
clf_transform.fit(X_transform_train,y_train)

print('original dataset test score : %f' %clf.score(X_test,y_test))
print('transform dataset test score : %f' %clf_transform.score(X_transform_test,y_test))


#包裹式特征选取：REFCV
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
estimator = LinearSVC()
selector = RFECV(estimator = estimator,cv=3)
selector.fit(X,y)
print('N_features : %s'%selector.n_features_)
print('Support is : %s'%selector.support_)
print('ranking is : %s'%selector.ranking_)
print('Grid Scores : %s'%selector.grid_scores_)


#嵌入式特征选取
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
estimator = LinearSVC(penalty='l1',dual=False)
selector = SelectFromModel(estimator=estimator,threshold='mean')
selector.fit(X,y)
selector.transform(X)
print('Threshold : %s'%selector.threshold_)
print('Support is : %s'%selector.get_support(indices=True))

#alpha和C与稀疏性的关系
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.datasets import load_iris,load_diabetes
%matplotlib

def test_Lasso(*data):
    X,y = data
    alphas = np.logspace(-2,2)
    zeros = []
    for alpha in alphas:
        regr = Lasso(alpha=alpha)
        regr.fit(X,y)
        num = 0
        for ele in regr.coef_:
            if abs(ele) <1e-5:
                num += 1
        zeros.append(num)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,zeros)
    ax.set_xlabel('alphas')
    ax.set_ylabel('zeros in coef')
    ax.set_xscale('log')

def test_LinearSVC(*data):
    X,y = data
    Cs = np.logspace(-2,2)
    zeros = []
    for C in Cs:
        regr = LinearSVC(C=C,penalty='l1',dual=False)
        regr.fit(X,y)
        num = 0
        for row in regr.coef_:
            for ele in row:
                if abs(ele) <1e-5:
                    num += 1
        zeros.append(num)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,zeros)
    ax.set_xlabel('Cs')
    ax.set_ylabel('zeros in coef')
    ax.set_xscale('log')

data = load_diabetes()
test_Lasso(data.data,data.target)
data = load_digits()
test_LinearSVC(data.data,data.target)

#学习器流水线，pipeline，将多个学习器组成流水线：数据标准化的学习器-->特征提取的流水线-->执行预测的学习器
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def test_Pipeline(data):
    X_train,X_test,y_train,y_test = data
    steps = [('Linear_SVM',LinearSVC(C=1,penalty='l1',dual=False))]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train,y_train)
    print('name steps : \n',pipeline.named_steps)
    print('Pipeline score : \n',pipeline.score(X_test,y_test))

data = load_digits()
X = data.data
y = data.target
test_Pipeline(model_selection.train_test_split(X,y,test_size=0.25,stratify=y))

#字典学习
from sklearn.decomposition import DictionaryLearning
X= [[1,2,3,4,5],
    [6,7,8,9,10],
    [10,9,8,7,6],
    [5,4,3,2,1]]
dct = DictionaryLearning(n_components=3)
dct.fit(X)
dct.transform(X)





