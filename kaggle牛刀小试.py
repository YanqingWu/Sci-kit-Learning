#kaggle - red_hat
import pandas as pd
import numpy as np
import pickle
import time
import os

#导入数据
#pd.set_option('display.max_columns', None)




#数据清理
people = pd.read_csv('/Users/wuyanqing/Downloads/red_hat/people.csv',parse_dates=['date'])
people.columns  #查看列名，看是否有特殊列，如date
people.set_index(keys=['people_id'],drop=True,append=False,inplace=True)

act_train = pd.read_csv('/Users/wuyanqing/Downloads/red_hat/act_train.csv',parse_dates=['date'])
act_train.set_index(keys=['people_id'],drop=True,append=False,inplace=True)

act_test = pd.read_csv('/Users/wuyanqing/Downloads/red_hat/act_test.csv',parse_dates=['date'])
act_test.set_index(keys=['people_id'],drop=True,append=False,inplace=True)

#合并数据
train_data = pd.merge(act_train,people,how='left',left_index=True,right_index=True,suffixes=('_act','_people'))
test_data = pd.merge(act_test,people,how='left',left_index=True,right_index=True,suffixes=('_act','_people'))

#拆分数据，官方解释type1和非type1的activity的格式式不同的
#查看有多少活动类型
train_data.columns
train_data.activity_category.value_counts()
#按照活动类型把数据分类
types = ['type %d'%i for i in range(1,8)]
train_datas = {}
test_datas = {}
for _type in types:
    train_datas[_type] = train_data[train_data.activity_category == _type].dropna(axis=(0,1),how='all')
    test_datas[_type] = test_data[test_data.activity_category == _type].dropna(axis=(0,1),how='all')
train_datas
#删除按照活动类型分类后的类型列，生成多级索引
for _type in types:
    train_datas[_type].drop('activity_category',axis=1,inplace=True)
    test_datas[_type].drop('activity_category',axis=1,inplace=True)
#去除唯一值，将activity_id设为行索引
for _type in types:
    train_datas[_type].set_index(keys = ['activity_id'],drop=True,append=True,inplace=True)
    test_datas[_type].set_index(keys = ['activity_id'],drop=True,append=True,inplace=True)

#类型转换
#查看数据类型
dtype_train = {}
dtype_test = {}
for _type in types:
    dtype_train[_type] = train_datas[_type].dtypes
    dtype_test[_type] = test_datas[_type].dtypes
dtype_train = pd.DataFrame(dtype_train)
dtype_test = pd.DataFrame(dtype_test)
dtype = pd.merge(dtype_train,dtype_test,left_index=True,right_index=True,suffixes=('_train','_test'))
#将所有数据转化为浮点数
#字符串列和bool值列的名字
str_col_list = ['group_1'] + ['char_%d_act'%i for i in range(1,11)] + ['char_%d_people'%i for i in range(1,10)]
bool_col_list = ['char_10_people'] + ['char_%d'%i for i in range(11,18)]
for _type in types:
    for data_set in [train_datas,test_datas]:
        data_set[_type].date_act = (data_set[_type].date_act - np.datetime64('1970-01-01')) / np.timedelta64(1,'D')
        data_set[_type].date_people = (data_set[_type].date_people - np.datetime64('1970-01-01'))/np.timedelta64(1,'D')
        data_set[_type].group_1 = data_set[_type].group_1.str.replace('group','').str.strip().astype(np.float64)
        for col in bool_col_list:
            if col in data_set[_type]:
                data_set[_type][col] = data_set[_type][col].astype(np.float64)
        for col in str_col_list[1:]:
            if col in data_set[_type]:
                data_set[_type][col] = data_set[_type][col].str.replace('type','').str.strip().astype(np.float64)
        data_set[_type] = data_set[_type].astype(np.float64)


        
        
        
        
        
#封装数据清理        
#根据前面的分析，给出一个Data_cleaner类，该类提供了load_data()方法，返回清洗好的数据
def current_time():
    '''
    以固定时间打印当前时间
    '''
    return time.strftime('%Y-%m-%d',time.localtime())
class Data_cleaner:
    '''
    数据清洗器
    它的初始化需要提供三个文件名，它的唯一对外接口为load_data()，返回清洗好的数据
    如果数据已存在则直接返回，否则执行一系列清洗操作，并返回清洗好的数据
    '''
    def __init__(self,people_file_name,act_train_file_name,act_test_file_name):
        '''
        :param people_file_name : people.csv的文件路径
        :param act_train_file_name : act_train_file_name的文件路径
        :param act_test_file_name : act_test_file_name的文件路径
        :return :
        '''
        self.p_fname = people_file_name
        self.train_fname = act_train_file_name
        self.test_fname = act_test_file_name
        self.types = ['type %d'%i for i in range(1,8)]
        self.fname = 'output/cleaned_data'
        
    def load_data(self):
        '''
        加载清洗好的数据
        如果数据已存在直接返回，如果不存在则加载csv文件，然后合并数据，拆分成type1-7，然后执行数据类型转换，最后重新排列每个列的顺序，保存并返回数据
        '''
        if(self._is_ready()):
            print('cleaned data is already availiabale ! \n')
            self._load_data()
        else:
            self._load_csv()
            self._merge_data()
            self._split_data()
            self._typecast_data()
            self._save_data()
            
    def _load_csv(self):
        '''
        加载CSV文件
        :return :
        '''
        print('----- Begin run load_csv at %s -----' %current_time()) 
        
        self.people = pd.read_csv(self.p_fname,sep='',header=0,keep_default_na=True,parse_dates=['date'])
        self.act_train = pd.read_csv(self.train_fname,sep='',header=0,keep_default_na=True,parse_dates=['date'])
        self.act_test = pd.read_csv(self.test_fname,sep='',header=0,keep_default_na=True,parse_dates=['date'])
        
        self.people.set_index(kesys = ['people_id'],drop=True,append=False,inplace=True)
        self.act_train.set_index(kesys = ['people_id'],drop=True,append=False,inplace=True)
        self.act_test.set_index(kesys = ['people_id'],drop=True,append=False,inplace=True)
        
        print('----- End run load_csv at %s -----' %current_time())
        
    def _merge_data(self):
        '''
        合并people数据和activity数据
        :return :
        '''
        
        print('----- Begin run merge_data at %s -----' %current_time())
        
        self.train_data = self.merge(act_train,people,how='left',left_index=True,right_index=True,suffixes=('_act','_people'))
        self.test_data = self.merge(act_test,people,how='left',left_index=True,right_index=True,suffixes=('_act','_people'))
        
        print('----- End run merge_data at %s -----' %current_time())
        
    def _split_data(self):
        '''
        拆分数据为 type1-7
        :return:
        '''
        
        print('----- Begin run split_data at %s -----' %current_time())
        
        self.train_datas = {}
        self.test_dats = {}
        for _type in self._types:
            #拆分
            self.train_datas[_type] = self.train_data[self.train_data.activity_category == _type].dropna(axis=(0,1),how='all')
            self.test_datas[_type] = self.test_data[self.test_data.activity_category == _type].dropna(axis=(0,1),how='all')
            #删除列activity_category
            self.train_datas[_type].drop('activity_category',axis=1,inplace=True)
            self.test_datas[_type].drop('activity_category',axis=1,inplace=True)
            #将列activity_id设为索引
            self.train_datas[_type].set_index(keys = ['activity_id'],drop=True,append=True,inplace=True)
            self.test_datas[_types].set_index(keys = ['activity_id'],drop=True,append=True,inplace=True)
        
        print('----- End run split_data at %s -----' %current_time())
            
    def _typecast_data(self):
        '''
        执行数据类型转换，将所有数据转换为浮点型
        :return :
        '''
        
        print('----- Begin run typecast_data at %s -----' %current_time())
        
        str_col_list = ['group_1'] + ['char_%d_act'%i for i in range(1,11)] + ['char_%d_people'%i for i in range(1,10)]
        bool_col_list = ['char_10_people'] + ['char_%d'%i for i in range(11,18)]
            
        for _type in sel.types:
            for data_set in [train_datas,test_datas]:
                data_set[_type].date_act = (data_set[_type].date_act - np.datetime64('1970-01-01')) / np.timedelta64(1,'D')
                data_set[_type].date_people = (data_set[_type].date_people - np.datetime64('1970-01-01'))/np.timedelta64(1,'D')
                data_set[_type].group_1 = data_set[_type].group_1.str.replace('group','').str.strip().astype(np.float64)
            for col in bool_col_list:
                if col in data_set[_type]:
                    data_set[_type][col] = data_set[_type][col].astype(np.float64)
            for col in str_col_list[1:]:
                if col in data_set[_type]:
                data_set[_type][col] = data_set[_type][col].str.replace('type','').str.strip().astype(np.float64)
            data_set[_type] = data_set[_type].astype(np.float64)

        print('----- Begin run typecast_data at %s -----' %current_time()) 
        
    def _is_ready(self):
        if (os.path.exists(self.fname)):
            return True
        else:
            return False
    
    def _save_data(self):
        
        print('----- Begin run save_data at %s -----' %current_time())
        
        with open (self.fname,'wb') as file:
            pickle.dump([self.train_datas,self.test_datas],file=file)
            
        print('----- End run save_data at %s -----' %current_time()')
            
    def _load_data(self):
        
        print('----- Begin run load_data at %s -----' %current_time())
              
        with open(self.fname,'rb') as file:
              self.train_datas,self.test_datas = pickle.load(file)
              
        print('----- End run load_data at %s -----' %current_time())
            
  
              
              
#数据预处理
#考察各列的取值集合
lambda_len = lambda x : len(x.unique())           
lambda_data = lambda x : str(x.unique()) if (len(x.unique())<=5) else str(x.unique()[:3]) + '...'          
train_results = {}
test_results = {}
types = ['type %d'%i for i in range(1,8)]
for _type in types:
    train_results[_type] = pd.DataFrame({'len':train_datas[_type].apply(lambda_len),
                                        'data':train_datas[_type].apply(lambda_data)},
                                        index = train_datas[_type].columns)
              
    test_results[_type] = pd.DataFrame({'len':test_datas[_type].apply(lambda_len),
                                        'data':test_datas[_type].apply(lambda_data)},
                                        index = test_datas[_type].columns)              
           
len_train = {}
data_train = {}
len_test = {}
data_test = {}
for _type in types:
    len_train[_type] = train_datas[_type].apply(lambda_len)
    data_train[_type] = train_datas[_type].apply(lambda_data)
    len_train_pd = pd.DataFrame(len_train)
    data_train_pd = pd.DataFrame(data_train)
    
    len_test[_type] = test_datas[_type].apply(lambda_len)
    data_test[_type] = test_datas[_type].apply(lambda_data)
    len_test_pd = pd.DataFrame(len_test)
    data_test_pd = pd.DataFrame(data_test)

train_results_pd = pd.merge(len_train_pd,data_train_pd,how='left',left_index=True,right_index=True,suffixes=('_len','_data'))   
train_results_pd.sort_index(axis=1)

test_results_pd = pd.merge(len_test_pd,data_test_pd,how='left',left_index=True,right_index=True,suffixes=('_len','_data'))   
test_results_pd.sort_index(axis=1)
#对于char_1_act - char_9_act这些列使用独热码，char_1_people - char_9_people也使用独热码，char_38位连续变量，group_1和char_10_act理论上需要独热码，考虑到曲直集合非常大，一旦进行独热编码，特征数量呈爆炸式增长，为了计算方便不进行独热编码
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import OneHotEncoder
from numpy import nan as NA

def onehot_encoder(train_datas,test_datas):
    train_results_onehot = {}
    test_results_onehot = {}
    types = ['type %d'%i for i in range(1,8)]
    for _type in types:
        if _type == 'type 1':
            one_hot_cols = ['char_%d_act'%i for i in range(1,10)] + ['char_%d_people'%i for i in range(1,10)]
            train_end_cols = ['group_1','date_act','date_people','char_38','outcome']
            test_end_cols = ['group_1','date_act','date_people','char_38']
        else:
            one_hot_cols = ['char_%d_people'%i for i in range(1,10)]
            train_end_cols = ['group_1','char_10_act','date_act','date_people','char_38','outcome']
            test_end_cols = ['group_1','char_10_act','date_act','date_people','char_38']
        
        train_front_array = train_datas[_type][one_hot_cols].values
        train_end_array = train_datas[_type][train_end_cols].values
        train_middle_array = train_datas[_type].drop(train_end_cols + one_hot_cols,axis=1,inplace=False).values
        
        test_front_array = test_datas[_type][one_hot_cols].values
        test_end_array = test_datas[_type][test_end_cols].values
        test_middle_array = test_datas[_type].drop(test_end_cols + one_hot_cols,axis=1,inplace=False).values

        encoder = OneHotEncoder(categorical_features='all',sparse=True)
        train_result = hstack([encoder.fit_transform(train_front_array),csr_matrix(train_middle_array),csr_matrix(train_end_array)])
        test_result = hstack([encoder.fit_transform(test_front_array),csr_matrix(test_middle_array),csr_matrix(test_end_array)])
        
        train_results_onehot[_type] = train_result
        test_results_onehot[_type] = test_result
              
    return train_results_onehot,test_results_onehot
onehot_encoder(train_datas,test_datas)

print('before encoder:\n')

for _type in types:
    print('train(type = %s) shape :'%_type,train_datas[_type].shape)
    print('test(type = %s) shape :'%_type,test_datas[_type].shape)

print('==========================\n\n')

train_results_onehot,test_results_onehot = onehot_encoder(train_datas,test_datas)

print('after encode:\n')

for _type in types:
    print('train(type = %s) shape :'%_type,train_results_onehot[_type].shape)
    print('test(type = %s) shape :'%_type,test_results_onehot[_type].shape)
              
print('==========================\n\n')
              
              
              
              

#归一化处理，经过独热编码后，列名已经没有了，上面独热编码后，没有经过独热编码的放在后几列
from sklearn.preprocessing  import MaxAbsScaler
def scale(train_datas,test_datas): 
    train_results={}
    test_results={}
    types=['type %d'%i for i in range(1,8)]
    
    for _type in types:
        if _type=='type 1':
            train_last_index=5#最后5列为 group_1/date_act/date_people/char_38/outcome
            test_last_index=4#最后4列为 group_1/date_act/date_people/char_38 
        else:
            train_last_index=6#最后6列为 group_1/char_10_act/date_act/date_people/char_38/outcome
            test_last_index=5#最后5列为 group_1/char_10_act/date_act/date_people/char_38 
        
        scaler=MaxAbsScaler()
        train_array=train_datas[_type].toarray()        
        train_front=train_array[:,:-train_last_index]
        train_mid=scaler.fit_transform(train_array[:,-train_last_index:-1])#outcome 不需要归一化
        train_end=train_array[:,-1].reshape((-1,1)) #outcome
        train_results[_type]=np.hstack((train_front,train_mid,train_end))
        
        test_array=test_datas[_type].toarray()
        test_front=test_array[:,:-test_last_index]
        test_end=scaler.transform(test_array[:,-test_last_index:])
        test_results[_type]=np.hstack((test_front,test_end))

    return train_results,test_results

train_datas,test_datas = onehot_encoder(train_datas,test_datas)
              
ta_results,tt_results=scale(train_results,test_results)
types=['type %d'%i for i in range(1,8)]
for _type in types:
    print("Train(type=%s):"%_type,np.unique(ta_results[_type].max(axis=1)),np.unique(ta_results[_type].min(axis=1)))
    print("Test(type=%s):"%_type,np.unique(tt_results[_type].max(axis=1)),np.unique(tt_results[_type].min(axis=1)))
              
              

              
              


              
import os
import pickle
import numpy as np
from scipy.sparse import hstack,scr_matrix
from sklearn.preprocessing import OneHotEncoder,MaxxAbsScaler
from data_clean import current_time
from data_clean import Data_Cleaner

class Data_preprocessing:
    '''
    数据预处理器
    它的初始化需要提供清洗好的数据，提供了唯一对外窗口，load_data()，返回预处理好的数据。
    如果数据已经存在，则直接返回，否则执行一系列预处理，并返回预处理好的数据。
    '''
    
    def __init__(self,train_datas,test_datas):
        '''
        :param train_datas:清洗好的训练集
        :param test_datas:清洗好的测试剂
        :return:
        '''
        self.types = train_datas.keys()
        self.train_datas = train_datas
        self.ytest_datas = test_datas
        
        self.fname = 'output/processed_data'
    def load_data(self):
        '''
        加载预处理好的数据，如果数据已存在则直接返回
        不存在则预处理数据，存储之后返回
        :return: 一个元组，依次为 : train_datas,test_datas
        '''
        if (self._is_ready()):
            print('preprocessed data is already availiable')
            self._load_data()
        else:
            self._onehot_encode()
            self._scaled()
            self._save_data()
    def _onehot_encode():
        '''
        独热编码
        :return :
        '''
        print('----- Begin run onehot_encoder at %s -----'%current_time())
        train_results = {}
        test_results = {}
        self._encoders = {}
        
        for _type in self.types():
            if _type == 'type 1':
                one_hot_cols = ['char_%d_act'%i for i in range(1,10)] + ['char_%d_people'%i for i in range(1,10)]
                train_end_cols = ['group_1','date_act','date_people','char_38','outcome']
                test_end_cols = ['group_1','date_act','date_people','char_38']
            else:
                one_hot_cols = ['char_%d_people'%i for i in range(1,10)]
                train_end_cols = ['group_1','char_10_act','date_act','date_people','char_38','outcome']
                test_end_cols = ['group_1','char_10_act','date_act','date_people','char_38']
            
            train_front_array = self.train_datas[_type][one_hot_cols].values
            train_end_array = self.train_datas[_type][train_end_cols].values
            train_middle_array = self.train_datas[_type].drop(train_end_cols + one_hot_cols,axis=1,inplace=False).values
            
            test_front_array = self.test_datas[_type][one_hot_cols].values
            test_end_array = self.test_datas[_type][test_end_cols].values
            test_middle_array = self.test_datas[_type].drop(test_end_cols + one_hot_cols,axis=1,inplace=False).values           
              
            encoder = OneHotEncoder(categorical_features='all',sparse=True)
            
            train_result = hstack([encoder.fit_transform(train_front_array),csr_matrix(train_middle_array),csr_matrix(train_end_array)])
            test_result = hstack([encoder.fit_transform(test_front_array),csr_matrix(test_middle_array),csr_matrix(test_end_array)])
            
            train_results[_type] = train_result
            test_results[_type] = test_result
        
        self.train_datas = train_results
        self.test_datas = test_results
        print('----- End run onehot_encoder at %s -----'%current_time())
    def _scaled(self):
        '''
        特征归一化，采用 MaxAbsScale 来进行归一化
        :return:
        '''
        print('----- Begin run Scaled at %s -----'%current_time())
        train_scales = {}
        test_scales = {}
        self.scaler = {}
        for _type in self.types:
            if _type == 'type 1':
                train_last_index = 5
                test_last_index = 4
            else:
                train_last_index = 6
                test_last_index = 5
            scaler = MaxAbsScaler()
            train_array = self.train_datas[_type].toarray()
            train_front = train_array[:,:-train_last_index]
            train_middle = scaler.fit_transform(train_array[:,-train_last_index:-1])
            train_end = train_array[:,-1].reshape((-1,1))
            train_scalers[_type] = np.hstack((train_front,train_middle,train_end))
            self.scaler[_type] = scaler
        
            test_array = self.test_datas[_type].toarray()
            test_front = test_array[:,:-test_last_index]
            test_end = scaler.fit_transform(test_array[:,-test_last_index])
            test_scales[_type] = np.hstack((test_front,test_end))
            self.scalers[_type] = scaler
            
              
        self.train_datas = train_scalers
        self.test_datas = test_scalers
              
    def _is_ready(self):
        if (os.path.exists(self.fname)):
            return True
        else:
            return False
    def _save_data(self):
        print('----- Begin run save_data at %s -----'%current_time())
        with open(self.fname,'wb') as file:
            pickle.dump([self.train_datas,self.test_datas,self.encoders(),self.scalers],file)
        print('----- End run save_data at %s -----'%current_time()')
    def _load_data():
        print('----- Begin run with _load_data at %s -----'%current_time())
        with open (self.fname,'rb') as file:
            self.train_datas,self.test_datas,self.encoders,self.scalers = pickle.load(file)
        print('----- End run with _load_data at %s -----'%current_time()')
    

              

              
              
#学习曲线和验证曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve,validation_curve

from sklearn.model_selection import train_test_split 
from data_clean import curent_time
from data_preprocessor import Data_preprocessor,Data_Cleaner

class Curve_Helper:
    '''
    学习曲线和验证曲线
    '''
    def __init__(self,curve_name,xlabel,x_islog):
        '''
        初始化函数
        :param curve_name:曲线名称
        :param xlabel:曲线X轴的名称
        :param x_islog：曲线X轴是否为对数
        :return:
        '''
        self.curve_name = curve_name
        self.xlabel = xlabel
        self.x_islog = x_islog
        def save_curve(self,x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std):
            '''
            保存曲线的数据
            :param x_data:曲线的x轴数据，也就是被考察的指标的序列
            :param train_scores_mean:训练集的平均得分
            :param train_scores_std:训练集得分的标准差
            :param test_scores_mean:测试集的平均得分
            :param test_scores_std:测试集得分的标准差
            :return：
            '''
            with open('output/%s'%self.curve_name,'wb') as output:
                result_array = np.array([x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std])
                np.save(output,result_array)
        def plot_curve(self,x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std):
            '''
            绘图并保存图片
            :param x_data:曲线的x轴数据，也就是被考察的指标的序列
            :param train_scores_mean:训练集的平均得分
            :param train_scores_std:训练集得分的标准差
            :param test_scores_mean:测试集的平均得分
            :param test_scores_std:测试集得分的标准差
            :return
            '''
              
            min_y1 = np.min(train_scores_mean)
            min_y2 = np.min(test_scores_mean)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(x_data,train_scores_mean,label = 'Training roc_auc',color = 'r',marker='o')
            ax.fill_between(x_data,train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,alpah=0.2,color='r')
            ax.plot(x_data,test_scores_mean,color='g',marker='+')
            ax.fill_between(x_data,test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,alpha=0.2,color='g')
            ax.set_title('%s'%self.curve_name)
            ax.set_xlabel('%s'%self.x_label)
            ax.locator_params(axis='x',tight=True,nbins=10)
            ax.grid(which='both')
            if self.x_islog:ax.set_xscale('log')
            ax.set_ylabel('Score')
            ax.set_ylim(min(min_y1,min_y2))
            ax.set_xlim(0,max(x_data))
            ax.legend(loc='best')
            ax.grid(True,which='both',axis='both')
            fig.savefig('output/%s.png':%self.curve_name,dpi=100)
            
        @classmethod
        def plot_from_saved_data():
            '''
            通过保存的数据点来绘制并保存图形
            
            :param file_name:保存数据点的文件名
            :param curve_name:曲线名称
            :param x_label:曲线X轴的名称
            :param x_islog:曲线X轴是否为对数
            '''
            
            x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std = np.load(file_name)
            helper = Curve_Helper(curve_name,xlabel,x_islog)
            helper.plot_curve(x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std)
        
        def cut_data(data,scale_factor,stratify=True):
            '''
            切分数据集，使用其中一部分来学习
            
            :param data:原始数据集
            :param stratify:传递给train_test_split的stratify参数
            :param scale_factor:传递给train_test_split的train_size参数
            '''
            
            if stratify:
                return train_test_split(data,train_size=sclae_factor,stratify=data[:-1])
            else :
                return train_test_split(data,train_size=sclae_factor)
        

              
              
    class Curver:
        '''
        用于生成学习曲线和验证曲线
        '''
              
        def create_curve(self,train_data,curve_name,xlabel,x_islog,sclae=0.1,is_gui=False):
            '''
            生成曲线
            
            :param train_data:训练数据集
            :param curve_name:曲线名称，用于绘图和保存文件
            :param xlabel:曲线X轴的名称
            :param x_islog:X轴是否为对数坐标
            :param scale:切分比例，默认使用10%的训练集
            :param is_gui:是否在gui环境下运行，如果在则绘制图片并保存
            :return :
            '''
            
            class_name = self.__class__.__name__
            self.curve_name = curve_name
            
            data = cut_data(train_data,scale_factor = scale,stratify=True)
            self.X = data[:,:-1]
            self.y = data[:,-1]
            
            result = self._curve()
            self.helper = Curve_Helper(self.curve_name,xlabel,x_islog)
            if (is_gui):
                self.helper.plot_curve(*result)
            self.helper.save_curve(*result)
            
class Learning_Curver(Curver):
    def __init__(self,train_sizes):
        self.train_sizes = train_sizes
        self.estimator = GradientBoostingClassifier(max_depth=10)
    
    def _curve(self):
        print('----- Begin run learning_curve (%s) at %s -----' %(self.curve_name,current_time()))
        abs_trains_sizes,train_scores,test_scores = learning_curve(self.estimator,self.X,self.y,cv=3,scoring='roc_auc',train_sizes=self.train_sizes)
        print('----- End run learning_curve (%s) at %s -----' %(self.curve_name,current_time())')
        train_scores_mean = np.mean(train_scores,axis=1)
        train_scores_std = np.std(train_scores,axis=1)
        test_scores_mean = np.mean(test_scores_mean)
        test_scores_std = np.std(test_scores_std)
        return abs_trains_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std
    
class ValidationCurve(Curver):
    def __init__(self,param_name,param_range):
        self.p_name = param_name
        self.p_range = param_range
        self.estimator = GradientBoostingClassifier()
              
    def _curve(self):
        print('----- Begin run validation_curve (%s) at %s'%(self.curve_name,curent_time())
        train_scores,test_scores = validation_curve(self.estimator,self.X,self.y,param_name=self.p_name,param_range=self.param_range,cv=3,scoring='roc_auc')
        print('----- End run validation_curve (%s) at %s'%(self.curve_name,curent_time()))
        train_scores_mean = np.mean(train_scores,axis=1)
        train_scores_std = np.std(train_scores,axis=1)
        test_scores_mean = np.mean(test_scores_mean)
        test_scores_std = np.std(test_scores_std)
        return [item for item in self.p_range],train_scores_mean,train_scores_std,test_scores_mean,test_scores_std
        
    def run_learning_curve(data,type_name):
        '''
        生成学习曲线
        :param data:训练集
        :param type_name:数据种类名
        :return:
        '''
        learning_curver = Learning_Curver(train_sizes=np.logspace(-1,0,endpoint=True,num=10,dtype='float'))
        learning_curver.create_curve(data,'learning_curve_%s'%type_name,xlabel='Nums',x_islog=True,scale=0.99,is_gui=True)
        
    def run_test_subsample(data,type_name,scale,param_range):
        '''
        生成验证曲线，验证subsample参数
        :param data:训练集
        :param type_name:数据种类名
        :param scale:样本比例，一个小于1.0的浮点值
        :param param_range:参数的范围
        :return :
        '''
        validation_curver = ValidationCurve('subsample',param_range=param_range)
        validation_curver.create_curve(data,'validation_curve_subsample_%s'%type_name,xlabel='subsample',x_islog=False,scale = scale,is_gui=True)
        
    def run_test_n_estimators(data,type_name,scale,subsample,param_range):
        '''
        生成验证曲线，验证n_estimators参数
        :param data:训练集
        :param type_name:数据种类名
        :param scale:样本比例，一个小于1.0的浮点值
        :param subsample: subsample参数
        :param param_range:n_estimator参数的范围
        :return:
        '''
        validation_curver = ValidationCurve('n_estimators',param_range=param_range)
        validation_curver.estimator.set_params(subsample=subsample)
        validation_curver.creat_curve(data,'validation_curve_n_estimators_%s'%type_name,xlabel='n_estimators',x_islog=True,scale=scale,is_gui=True)
    
    def run_test_maxdepth(data,type_name,scale,subsample,n_estimators,param_range):
        '''
        生成验证曲线，验证maxdepth参数
        :param data:训练集
        :param type_name:数据种类名
        :param scale:样本比例，一个小于1.0的浮点值
        :param subsample:subsample参数
        :param n_estimators:n_estimators参数
        :param param_range:参数范围
        '''
        validation_curver = ValidationCurve('max_depth',param_range=param_range)
        validation_curver.estimator.set_params(subsample=subsammple)
        validation_curver.estimator.set_params(n_estimators=n_estimators)
        validation_curver.create_curve(data,'validation_curve_maxdepth_%s'%type_name,xlabel='maxdepth',x_islog=True,scale=scale,is_gui=True)
    
              
              
              
              
              
              
              
