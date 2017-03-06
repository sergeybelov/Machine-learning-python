# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:59:46 2017


"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#==============================================================================
# Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true)
# и ответы некоторого классификатора (колонка predicted).
#==============================================================================
data= pd.read_csv('classification.csv')


#Заполните таблицу ошибок классификации
#==============================================================================
# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.
#==============================================================================
classificationErrorMatrix=np.array([[0,0],[0,0]])
trueCol=0
predCol=1

for dt in data.values:
    if dt[trueCol]==1:
        if dt[predCol]==1: classificationErrorMatrix[0][0]+=1
        else: classificationErrorMatrix[1][0]+=1
    else:
        if dt[predCol]==1: classificationErrorMatrix[0][1]+=1
        else: classificationErrorMatrix[1][1]+=1

print ('Таблица ошибок')
print (' '.join(map(lambda x: str(x[0])+' '+str(x[1]), classificationErrorMatrix)))
#==============================================================================
# Посчитайте основные метрики качества классификатора:
# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
# Precision (точность) — sklearn.metrics.precision_score
# Recall (полнота) — sklearn.metrics.recall_score
# F-мера — sklearn.metrics.f1_score
#==============================================================================
accuracy_values=[]
accuracy_values.append(round(accuracy_score(data['true'].values,data['pred'].values),2))
accuracy_values.append(round(precision_score(data['true'].values,data['pred'].values),2))
accuracy_values.append(round(recall_score(data['true'].values,data['pred'].values),2))
accuracy_values.append(round(f1_score(data['true'].values,data['pred'].values),2))

print ('Оценки качества')
print (' '.join(map(lambda x: str(x), accuracy_values)))


del classificationErrorMatrix
del accuracy_values

#==============================================================================
# Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.
#==============================================================================
data= pd.read_csv('scores.csv')
