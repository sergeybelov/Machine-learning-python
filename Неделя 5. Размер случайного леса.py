# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:13:52 2017


"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from  sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
#==============================================================================
# Загрузите данные из файла abalone.csv.
# Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
#==============================================================================
df = pd.read_csv('abalone.csv')

#==============================================================================
# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
# Если вы используете Pandas, то подойдет следующий код:
# data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
#==============================================================================
df.Sex=df.Sex.map({'M' : 1, 'F' : -1, 'I':0})

#Разделите содержимое файлов на признаки и целевую переменную.
#В последнем столбце записана целевая переменная, в остальных — признаки.
train_X=df.iloc[:,:-1].values
train_Y=df.iloc[:,-1:].values.ravel()

#==============================================================================
# Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с
# различным числом деревьев: от 1 до 50 (не забудьте выставить "random_state=1" в конструкторе).
# Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам.
#
# Используйте параметры "random_state=1" и "shuffle=True" при создании генератора кросс-валидации sklearn.cross_validation.KFold.
# В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
#==============================================================================
ms=make_scorer(r2_score)#В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
validationTest={}#инициализируем словарь

kf = KFold(random_state=1,shuffle=True)#Конструктор кросс-валидации
for N in range(50):
    n_estimators=N+1
    regressor = RandomForestRegressor(n_estimators,random_state=1)#Конструктор случайного леса
    #обучать алгоритм не нужно, мы лишь проверяем его качество

    scores = cross_val_score(regressor, train_X, train_Y, scoring=ms ,cv=kf)#Оценка алгоритма
    val=scores.mean()#берем среднее значение оценки
    if val>0.52:#проверка качества по целевому показателю
        validationTest[n_estimators]=val

#==============================================================================
# Определите, при каком минимальном количестве деревьев случайный лес показывает качество
# на кросс-валидации выше 0.52. Это количество и будет ответом на задание.
#==============================================================================
#формируем датасет для сортировки
validationTestDataFrame=pd.DataFrame.from_dict(validationTest, orient='index')#получаем из словаря датасет
validationTestDataFrame.index.name = 'k'
validationTestDataFrame.columns =['Scores']
validationTestDataFrame.sort_index(ascending=True,inplace=True)
print('Определите, при каком минимальном количестве деревьев случайный лес показывает качество ');
print(validationTestDataFrame.head(5))


