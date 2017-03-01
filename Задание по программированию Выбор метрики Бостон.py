# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:11:38 2017

@author: нзнегз
"""
import pandas
import numpy
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). 
#Результатом вызова данной функции является объект, у которого признаки 
#записаны в поле data, а целевой вектор — в поле target.
boston=datasets.load_boston()

# - CRIM     per capita crime rate by town
# - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS    proportion of non-retail business acres per town
# - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - NOX      nitric oxides concentration (parts per 10 million)
# - RM       average number of rooms per dwelling
# - AGE      proportion of owner-occupied units built prior to 1940
# - DIS      weighted distances to five Boston employment centres
# - RAD      index of accessibility to radial highways
# - TAX      full-value property-tax rate per $10,000
# - PTRATIO  pupil-teacher ratio by town
# - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT    % lower status of the population
# - MEDV     Median value of owner-occupied homes in $1000's
data=pandas.DataFrame(data=boston.data,columns=boston.feature_names)
target=pandas.DataFrame(data=boston.target)
boston=0

#Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
dataScaled=scale(data)#масштабирование выполняется перед обучением
data=0

#Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, 
#чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace). 
#Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, 
#зависящие от расстояния до ближайших соседей. 
#В качестве метрики качества используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' у cross_val_score; 
#при использовании библиотеки scikit-learn версии 18.0.1 и выше необходимо указывать scoring='neg_mean_squared_error'). 
#Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42, 
#не забудьте включить перемешивание выборки (shuffle=True).
kf = KFold(n_splits=5, shuffle=True, random_state=42)#кросс-валидация по 5 блокам с random_state = 42 с перемешиванием выборки

#Полный формат вызова функции: numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False),
#где:
#start - обязательный аргумент, первый член последовательности элементов массива;
#stop - обязательный аргумент, последний член последовательности элементов массива;
#num - опциональный аргумент, количество элементов массива, по умолчанию равен 50;
#endpoint - опциональный аргумент, логическое значение, по умолчанию True. 
#Если передано True, stop, последний элемент массива. 
#Если установлено в False, последовательность элементов формируется от start до stop для num + 1 элементов, 
#при этом в возвращаемый массив последний элемент не входит;
#retstep - опциональный аргумент, логическое значение, по умолчанию False. 
#Если передано True, функция возвращает кортеж из двух членов, 
#первый - массив, последовательность элементов, второй - число, приращение между элементами последовательности.
p_parameter=numpy.linspace(1, 10, num = 200, endpoint = True, retstep = False)

validationTest={}
for parameter in p_parameter:#организуем цикл по p    
    #Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, 
    #зависящие от расстояния до ближайших соседей. 
    regression = KNeighborsRegressor(n_neighbors=5, weights='distance', p=parameter)
    regression.fit(dataScaled,target)
    scores=cross_val_score(regression, dataScaled, target, scoring='neg_mean_squared_error',cv=kf)
    validationTest[parameter] = round(scores.mean(),1)
    
    
#формируем датасет для сортировки    
validationTestDataFrame=pandas.DataFrame.from_dict(validationTest, orient='index')#получаем из словаря датасет  
validationTestDataFrame.index.name = 'p'
validationTestDataFrame.columns =['Scores']
validationTestDataFrame.sort_values(['Scores'], ascending=[False],inplace=True)#сортировка по убыванию значений
print('Определите, при каком p качество на кросс-валидации оказалось оптимальным.');
print(validationTestDataFrame.head(1))
    