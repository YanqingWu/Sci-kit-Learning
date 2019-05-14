import pandas as pd

import matplotlib.pyplot as plt

%matplotlib

titanic = pd.read_csv('/Users/wuyanqing/Downloads/train.csv')

titanic.describe()  #审阅数据

titanic.isnull().sum()  #查看缺失值

titanic.Age.fillna(titanic.Age.median(),inplace = True) #替换缺失值为中位数

titanic.Sex.value_counts()  #性别的汇总统计
 
#对性别进行分析

    survived = titanic[titanic.Survived == 1].Sex.value_counts()  #生还者中男女的人数

    dead = titanic[titanic.Survived == 0].Sex.value_counts()  #未生还者中男女的人数

    df = pd.DataFrame([survived,dead],index = ('survived','dead'))

    df.plot.bar()  # 生存和死亡中男女条形图

    df = df.T   #行列替换

    df.plot.bar()  #男性和女性中存活和死亡数条形图

    df.plot(kind = 'bar',stacked = True)

    df['p_survived'] = df.survived/(df.survived + df.dead)  #性别存活死亡比

    df['p_dead'] = df.dead/(df.survived + df.dead)

    df[['p_survived','p_dead']].plot.bar(stacked = True)  #同一性别生存死亡比

#对年龄进行分析

    survived = titanic[titanic.Survived == 1].Age  #生还者中男女的人数

    dead = titanic[titanic.Survived == 0].Age #未生还者中男女的人数

    df = pd.DataFrame([survived,dead],index = ('survived','dead'))

    df = df.T   #行列替换  

    df.plot.hist(stacked = True)  #直方图
    
    df.plot.kde(xlim = (0,80))  #  死亡和生存密度图

    age = 16
    
    young = titanic[titanic.Age <= age]['Survived'].value_counts()
    
    old = titanic[titanic.Age > age]['Survived'].value_counts()
    
    df = pd.DataFrame([young,old],index = ['young','old'])
    
    df.columns = ['dead','survived']
    
    df.plot.bar(stacked = True)

#对票价进行分析
    
    survived = titanic[titanic.Survived == 1].Fare  #生还者中男女的人数

    dead = titanic[titanic.Survived == 0].Fare #未生还者中男女的人数

    df = pd.DataFrame([survived,dead],index = ('survived','dead'))

    df = df.T   #行列替换  

    df.plot.hist(stacked = True)  #直方图
    
    df.plot.kde( xlim =(0,513) )  #  死亡和生存密度图

################同时查看年龄和票价的影响################
    age = titanic[titanic.Survived == 0].Age
    
    fare = titanic[titanic.Survived == 0].Fare
    
    plt.scatter(age, fare, s=10, alpha=0.5)
    
    age = titanic[titanic.Survived == 1].Age
    
    fare = titanic[titanic.Survived == 1].Fare
    
    plt.scatter(age, fare, s=10, alpha=0.5)
    

    
    










