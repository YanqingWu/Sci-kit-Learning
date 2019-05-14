#损失函数：zero_one_loss,返回误分类的比例或者个数
from sklearn.metrics import zero_one_loss
y_true = [1,1,1,1,1,1,1,1,0,0,0,0,0,0]
y_pred = [1,1,1,1,1,1,0,0,0,0,0,0,0,0]
zero_one_loss(y_true,y_pred,normalize=True)

#损失函数：对数损失函数
from sklearn.metrics import log_loss
y_true = [1,1,1,0,0,0]
y_pred = [
          [0.1,0.9],
          [0.2,0.8],
          [0.3,0.7],
          [0.2,0.8],
          [0.4,0.6],
          [0.5,0.5]
          ]
log_loss(y_true,y_pred,normalize=True)

#数据集切分
import numpy as np
from sklearn.model_selection import train_test_split
X = np.arange(20).reshape((5,4))
y = [1,1,1,0,0]
X_train,X_test,y_train,y_test = train_test_split(X,y)
X_train

#数据切分：KFold切分，StratifiedKFold分层采样
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
X = np.arange(32).reshape(8,4)
y = [1,1,0,0,1,1,0,0]
folder_1 = KFold(n_splits = 4,shuffle=True)
folder_1.split(X,y)
folder_2 = StratifiedKFold(n_splits=4,shuffle=True)
folder_2.split(X,y)

#数据切分：留一法，LeaveOneOut
from sklearn.model_selection import LeaveOneOut
X = np.arange(32).reshape(8,4)
y = [1,1,0,0,1,1,0,0]
lo = LeaveOneOut(len(y))

#cross_val_score，在指定数据集上运行指定学习器
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
digits = load_digits()
X = digits.data
y = digits.target
results = cross_val_score(LinearSVC(),X,y,cv=10)
print(results)

#分类问题的性能度量：学习器的.score方法，model_selection中的评估工具如cross_val_score，metric中的函数
#准确率
from sklearn.metrics import accuracy_score
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
accuracy_score(y_true,y_pred)

#查准率
from sklearn.metrics import precision_score
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
precision_score(y_true,y_pred)

#查全率
from sklearn.metrics import recall_score
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
recall_score(y_true,y_pred)

#f1_score
from sklearn.metrics import f1_score
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
f1_score(y_true,y_pred)

#fbeta_score
from sklearn.metrics import fbeta_score
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
fbeta_score(y_true,y_pred,beta=0.5)
fbeta_score(y_true,y_pred,beta=2)

#classfication_report，给出分类结果的主要性能指标
from sklearn.metrics import classification_report
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
print(classification_report(y_true,y_pred))

#confution_matrix
from sklearn.metrics import confusion_matrix
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
print(confusion_matrix(y_true,y_pred))

#precision_recall_curve，计算分类结果的P-R曲线
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
%matplotlib

iris = load_iris()
X = iris.data
y = iris.target
#二元化标记
y = label_binarize(y,classes=[0,1,2])
n_classes = y.shape[1]
#添加噪声
n_samples,n_features = X.shape
X = np.c_[X,np.random.randn(n_samples,200*n_features)]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
#训练模型
clf = OneVsRestClassifier(SVC(kernel='linear',probability = True))
clf.fit(X_train,y_train)
y_score = clf.fit(X_train,y_train).decision_function(X_test)
#绘制P-R
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i],recall[i],_ = precision_recall_curve(y_test[:,i],y_score[:,i])
    ax.plot(recall[i],precision[i],label = 'target = %s'%i)
ax.set_xlabel('Recall Score')
ax.set_ylabel('Precision Score')
ax.set_title('P-R')
ax.legend()
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.grid()




#绘制roc曲线
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
%matplotlib

iris = load_iris()
X = iris.data
y = iris.target
#二元化标记
y = label_binarize(y,classes=[0,1,2])
n_classes = y.shape[1]
#添加噪声
n_samples,n_features = X.shape
X = np.c_[X,np.random.randn(n_samples,200*n_features)]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
#训练模型
clf = OneVsRestClassifier(SVC(kernel='linear',probability = True))
clf.fit(X_train,y_train)
y_score = clf.fit(X_train,y_train).decision_function(X_test)
#获取ROC曲线
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
    roc_auc[i] = roc_auc_score(y_test[:,i],y_score[:,i])
    ax.plot(fpr[i],tpr[i],label = 'target = %s, auc = %s'%(i,roc_auc[i]))
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC')
ax.legend()
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.grid()



#回归问题的性能度量
#mean_absolute_error，预测误差绝对值的均值
from sklearn.metrics import mean_absolute_error
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
mean_absolute_error(y_true,y_pred)

#mean_squared_error，预测误差平方的平均值
from sklearn.metrics import mean_squared_error
y_true = [1,1,1,1,1,0,0,0,1,0]
y_pred = [0,0,0,1,1,1,1,1,0,0]
mean_squared_error(y_true,y_pred)




#验证曲线，给出了学习器因为某个参数的不同取值在同一个测试集上的预测曲线
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.model_selection import validation_curve

digits = load_digits()
X = digits.data
y = digits.target

param_name = 'C'
param_range = np.logspace(-2,2)
train_scores,test_scores = validation_curve(LinearSVC(),X,y,param_name=param_name,param_range=param_range,cv=10,
                                           scoring='accuracy')
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
train_scores_std = np.std(train_scores,axis=1)
test_scores_std = np.std(test_scores,axis=1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.semilogx(param_range,train_scores_mean,label='Training Scores',color = 'r')
ax.fill_between(param_range,train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,alpha = 0.2,color='r')
ax.semilogx(param_range,test_scores_mean,label='Testing Scores',color = 'g')
ax.fill_between(param_range,test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,alpha = 0.2,color='g')
ax.set_title('Validation Curve')
ax.set_xlabel('C')
ax.set_ylabel('Score')
ax.set_ylim(0,1.1)
ax.legend()

#学习曲线
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
%matplotlib

digits = load_digits()
X = digits.data
y = digits.target

train_sizes = np.linspace(0.01,1,endpoint=True)
abs_train_sizes,train_scores,test_scores = learning_curve(LinearSVC(),
                                                          X,y,cv=10,scoring='accuracy',train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
train_scores_std = np.std(train_scores,axis=1)
test_scores_std = np.std(test_scores,axis=1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(train_sizes,train_scores_mean,label='Training Scores',color = 'r')
ax.fill_between(train_sizes,train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,alpha = 0.2,color='r')
ax.plot(train_sizes,test_scores_mean,label='Testing Scores',color = 'g')
ax.fill_between(train_sizes,test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,alpha = 0.2,color='g')
ax.set_title('Learning Curve')
ax.set_xlabel('Train Size')
ax.set_ylabel('Score')
ax.set_ylim(0,1.1)
ax.legend()

#参数优化，暴力搜索GridSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify=y)
tuned_parameters = {'penalty':('l1','l2'),
                    'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                    'solver':('liblinear','lbfgs')}

clf = GridSearchCV(LogisticRegression(),tuned_parameters,cv=10)
clf.fit(X_train,y_train)
print('Best Parameters : \n',clf.best_params_)
print('Best Score : \n',clf.best_score_)
print('Optimized score : \n',clf.score(X_test,y_test))
y_true,y_pred = y_test,clf.predict(X_test)
print('Detailed Classification Report : \n',classification_report(y_true,y_test))

#随机搜索
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify=y)
tuned_parameters = {'penalty':('l1','l2'),
                    'C':[0.01,0.05,0.1,0.5,1,5,10,50,100]}

clf = RandomizedSearchCV(LogisticRegression(),tuned_parameters,cv=10)
clf.fit(X_train,y_train)
print('Best Parameters : \n',clf.best_params_)
print('Best Score : \n',clf.best_score_)
print('Optimized score : \n',clf.score(X_test,y_test))
y_true,y_pred = y_test,clf.predict(X_test)
print('Detailed Classification Report : \n',classification_report(y_true,y_test))





