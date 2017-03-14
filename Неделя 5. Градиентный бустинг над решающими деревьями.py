# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:54:57 2017


"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from math import exp
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#==============================================================================
# Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy (параметр values у датафрейма).
# В первой колонке файла с данными записано, была или нет реакция. Все остальные колонки (d1 - d1776)
# содержат различные характеристики молекулы, такие как размер, форма и т.д.
# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241.
#==============================================================================
data_train=pd.read_csv('gbm-data.csv').values

# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_train[:,1:],data_train[:,:1].ravel(),test_size = 0.8,random_state = 241)#X- начиная со второй колонки, Y - только первая колонка

del data_train

#==============================================================================
# Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 и
# для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
#==============================================================================
vfunc = np.vectorize(lambda y: 1/(1+exp(-y)))#настройка  функции для numpy array

def calc(X,y_true,n_est):
    score=np.empty(n_est)
    df=clf.staged_decision_function(X)# предсказания качества на обучающей и тестовой выборке на каждой итерации
    for i, y_pred in enumerate(df):
        score[i] = log_loss(y_true, vfunc(y_pred))#Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}), где y_pred — предсказанное значение. 1/(1+exp(-y))
    return score



learning_rate=[1, 0.5, 0.3, 0.2, 0.1]
n_est=250
for lr in learning_rate:
    clf=GradientBoostingClassifier(n_estimators=n_est, verbose=True, random_state=241,learning_rate=lr)
    clf.fit(X_train,y_train)

    train_score = calc(X_train,y_train,n_est)
    test_score = calc(X_test,y_test,n_est)

    if lr==0.2: test_score02=test_score

    plt.figure()
    plt.plot(test_score, 'r', linewidth=3)
    plt.plot(train_score, 'g', linewidth=2)
    plt.legend(['test', 'train'])

#Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации: переобучение
#(overfitting) или недообучение (underfitting)? В ответе укажите одно из слов overfitting либо underfitting.
#Ответ - underfitting


#==============================================================================
# 4. Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается, при learning_rate = 0.2.
#==============================================================================
test_scorePD=pd.DataFrame(data=test_score02)
test_scorePD.sort_values([0],inplace=True, ascending=True)
test_scorePD[0]=test_scorePD[0].map(lambda x: round(x,2))
print('Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается, при learning_rate = 0.2')
val=test_scorePD.head(1)
print(val)

#==============================================================================
# 5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций,
# на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и
# остальными параметрами по умолчанию.
#
# Какое значение log-loss на тесте получается у этого случайного леса?
# (Не забывайте, что предсказания нужно получать с помощью функции predict_proba.
# В данном случае брать сигмоиду от оценки вероятности класса не нужно)
#==============================================================================
clf = RandomForestClassifier(n_estimators=val.index.values[0],random_state=241)#Конструктор случайного леса
clf.fit(X_train,y_train)

pred = clf.predict_proba(X_test)
score=log_loss(y_test, pred)

print('Какое значение log-loss на тесте получается у этого случайного леса?')
print(round(score,2))