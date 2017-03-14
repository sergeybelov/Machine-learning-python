# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:54:57 2017


"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#==============================================================================
# Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy (параметр values у датафрейма).
# В первой колонке файла с данными записано, была или нет реакция. Все остальные колонки (d1 - d1776)
# содержат различные характеристики молекулы, такие как размер, форма и т.д.
# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241.
#==============================================================================
data_train=pd.read_csv('gbm-data.csv').values

# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_train[:,:-1],data_train[:,:1].ravel(),test_size = 0.8,random_state = 241)

del data_train

#==============================================================================
# Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 и
# для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
#==============================================================================
learning_rate=[1]#, 0.5, 0.3, 0.2, 0.1]
for lr in learning_rate:
    gbc=GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241,learning_rate=lr)
    gbc.fit(X_train,y_train)

    df=gbc.staged_decision_function(X_test)# предсказания качества на обучающей и тестовой выборке на каждой итерации