# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:44:28 2017

@author: нзнегз
"""
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


#Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv. 
#Целевая переменная записана в первом столбце, признаки — во втором и третьем.
data= pandas.read_csv('perceptron-train.csv',header=None)#,names=['Class','Feature1','Feature2']
y_train = data[0]#Класс
X_train = data.loc[:, 1:]#Характеристики

#target_train = data.Class#вычленяем массив
#model_train = data.drop(['Class'], axis=1)#Вычленяем модель из датасет 

data= pandas.read_csv('perceptron-test.csv',header=None)#,names=['Class','Feature1','Feature2']
y_test = data[0]#Класс
X_test = data.loc[:, 1:]#Характеристики

data=0

#процедура обучения перцептрона
def PerceptronTrain(typeStr,X_train, y_train,X_test,y_test):     
    #Обучите персептрон со стандартными параметрами и random_state=241.
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)

    #В качестве метрики качества мы будем использовать долю верных ответов (accuracy). 
    #Для ее подсчета можно воспользоваться функцией sklearn.metrics.accuracy_score, 
    #первым аргументом которой является вектор правильных ответов, а вторым — вектор ответов алгоритма.
    #Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
    predictions = clf.predict(X_test)#предсказание по тестовым данным
    accuracyScore=accuracy_score(y_test, predictions)#проверка точности по тестовым данным

    print('Подсчитайте качество '+typeStr)
    accuracyScoreRounded=round(accuracyScore,3)
    print(accuracyScoreRounded)
    return accuracyScoreRounded

#обучаем перцептрон
accuracyScoreGeneral=PerceptronTrain('без нормализации',X_train, y_train,X_test,y_test)

#Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
accuracyScoreScaled=PerceptronTrain('c нормализацией',X_train_scaled, y_train,X_test_scaled,y_test)

#Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее. Это число и будет ответом на задание.
Comparation=accuracyScoreScaled-accuracyScoreGeneral
print('Разница')
print(Comparation)