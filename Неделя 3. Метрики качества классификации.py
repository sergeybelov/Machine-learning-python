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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
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

#==============================================================================
# Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)?
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
#==============================================================================
def computeAUCROC(y_true,y_score):
    return roc_auc_score(y_true,y_score)

y_true=data['true'].values
accuracy_values={}
accuracy_values['score_logreg']=roc_auc_score(y_true,data['score_logreg'].values)
accuracy_values['score_svm']=roc_auc_score(y_true,data['score_svm'].values)
accuracy_values['score_knn']=roc_auc_score(y_true,data['score_knn'].values)
accuracy_values['score_tree']=roc_auc_score(y_true,data['score_tree'].values)

dt=pd.DataFrame(data=accuracy_values,index=[0]).transpose()
dt.sort_values([0], ascending=[False],inplace=True)
print('Наименование лучшего классификатора')
print(dt.head(1))
print('-----')
del accuracy_values

#==============================================================================
# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции
# sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
# В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds.
# Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
#==============================================================================
def CalculateThresholds(y_true,y_scores):
    precision, recall, thresholds=precision_recall_curve(y_true,y_scores)
    dt=pd.DataFrame(data=precision, columns=['precision'])#Создаем датафрейм
    dt['recall']=recall#добавляем колонку
    dt['precision_sel']=dt.apply(lambda x: 0 if x['recall']<0.7 else x['precision'],axis=1)#выкидываем лишние записи из точности по фильтру полноты
    dt.sort_values(['precision_sel'], ascending=[False],inplace=True)#сортируем по точности
    dt=dt.reset_index(drop=True)#реиндексируем
    return dt.loc[0,'precision_sel']#выбираем максимум точности

accuracy_values={}
accuracy_values['score_logreg']=CalculateThresholds(y_true,data['score_logreg'].values)
accuracy_values['score_svm']=CalculateThresholds(y_true,data['score_svm'].values)
accuracy_values['score_knn']=CalculateThresholds(y_true,data['score_knn'].values)
accuracy_values['score_tree']=CalculateThresholds(y_true,data['score_tree'].values)

dt=pd.DataFrame(data=accuracy_values,index=[0]).transpose()
dt.sort_values([0], ascending=[False],inplace=True)
print('Какой классификатор достигает наибольшей точности')
print(dt.head(1))
print('-----')