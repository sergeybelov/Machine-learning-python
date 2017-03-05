# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:44:48 2017


"""

import pandas as pd
from math import exp
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score


#==============================================================================
# Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой принимает значения -1 или 1.
#==============================================================================

data= pd.read_csv('data-logistic.csv',header=None)

#==============================================================================
# Реализуйте градиентный спуск (gradient descent) для обычной и
# L2-регуляризованной (с коэффициентом регуляризации 10) логистической регрессии - это коэффициент C.
# Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
#==============================================================================
def GradientDescent(C,data):
    print('C=%d' %C)
    errorAccuracy=10**-5
    weights=[0.,0.]
    k=0.1#длина шага
    l=data[0].count()#количество элементов в выборке
    distance_euclidean=0

    weightsDelta=[0.,0.]
    for step in range(10000):#Рекомендуется ограничить сверху число итераций десятью тысячами.
        oldweightsDelta=weightsDelta
        weightsDelta=[0.,0.]

        for obj in data.values:
            y=obj[0]
            gradient=y*(1-1/(1+exp(-y*(weights[0]*obj[1]+weights[1]*obj[2]))))
            weightsDelta=list(map(lambda w,wd,x: wd+x*gradient-k*C*w,weights,weightsDelta,obj[1:]))

        #доведите до сходимости (евклидово расстояние между векторами весов на соседних итерациях должно быть не больше 1e-5)
        distance_euclidean=distance.euclidean(weightsDelta,oldweightsDelta)
        if distance_euclidean<errorAccuracy:
            print('Дошли до заданной ошибки точности на шаге: %d' % step)
            break

        weights=list(map(lambda w,wd: w+wd*k/l,weights,weightsDelta))

    print('Евклидово расстояние=%.6f' %distance_euclidean)
    return weights

#==============================================================================
# Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
# Эти величины будут ответом на задание. В качестве ответа приведите два числа через пробел.
# Обратите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей, подсчитанные обученным алгоритмом.
# Для этого воспользуйтесь сигмоидной функцией: a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).
#==============================================================================
def GetAUC_ROC(C,data):
    weights=GradientDescent(C,data)
    y_true = list(map(lambda y: 0 if y<0 else 1 ,data[0].values))#Класс объекта
    X_train = data.loc[:, 1:]#Характеристики объектов

    y_scores=list(map(lambda x: 1 / (1 + exp(-weights[0]*x[0] - weights[1]*x[1])) ,X_train.values))
    aUC_ROC=roc_auc_score(y_true, y_scores)
    print ('AUC_ROC=%.3f' % round(aUC_ROC,3))
    print('-----')
    return round(aUC_ROC,3)

#выводим ответ
answ=[]
answ.append(GetAUC_ROC(0,data))
answ.append(GetAUC_ROC(2,data))

print ('Ответ')
print (' '.join(map(lambda x: str(x), answ)))