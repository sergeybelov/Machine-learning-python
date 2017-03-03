# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:44:48 2017


"""

import pandas as pd
from math import exp
from math import sqrt


#==============================================================================
# Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой принимает значения -1 или 1.
#==============================================================================

data= pd.read_csv('data-logistic.csv',header=None)
#y_train = data[0]#Класс объекта
#X_train = data.loc[:, 1:]#Характеристики объектов

#y_train.columns = ['x']
#X_train.columns = ['x1','x2']#xi1 и xi2 — значение первого и второго признаков соответственно на объекте xi
#data=None#Очистка памяти от переменной


#==============================================================================
# Реализуйте градиентный спуск (gradient descent) для обычной и
# L2-регуляризованной (с коэффициентом регуляризации 10) логистической регрессии - это коэффициент C.
# Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
#==============================================================================
#weight=pd.DataFrame({'w1' : [0.],
#                    'w2' : [0.]})#Инициализируем матрицу весов
def GradientDescent(C,data):
    errorAccuracy=10**-5
    weights=[0.,0.]
    k=0.1#длина шага
    l=data[0].count()#количество элементов в выборке

    weightsDelta=[0.,0.]
    for step in range(10000):#Рекомендуется ограничить сверху число итераций десятью тысячами.
        for obj in data.values:
            oldweightsDelta=weightsDelta

            weightsDelta=[0.,0.]
            mainValue=obj[0]*(1-1/(1+exp(-obj[0]*(weights[0]*obj[1]+weights[1]*obj[2]))))
            for index in range(2):
                weightsDelta[index]+=obj[index+1]*mainValue-k*C*weights[index]

        #доведите до сходимости (евклидово расстояние между векторами весов на соседних итерациях должно быть не больше 1e-5)
        _sum=0
        for index in range(2):#рассчитываем евклидово расстояние
            val=oldweightsDelta[index]-weightsDelta[index]
            _sum+=val**2
        if sqrt(_sum)<errorAccuracy: break#выходим из цикла если достигли целевого показателя ошибки

        #for index in range(2):
        #    weights[index]+=weightsDelta[index]*k/l
        # --- Эквивалентно ---
        #weights=map(lambda w,wd: w+wd*k/l,weights,weightsDelta)
        weights=[w+wd*k/l for w,wd in zip(weights,weightsDelta)]


    return weights

weights=GradientDescent(1,data)#C=1
