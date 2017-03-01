# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:27:46 2017

@author: нзнегз
"""

#Характеристики вин
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier

#Загрузите выборку Wine по адресу 
#https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data (файл также приложен к этому заданию)
data = pandas.read_csv('wine.data')

#Извлеките из данных признаки и классы. Класс записан в первом столбце 
#(три варианта), признаки — в столбцах со второго по последний. 
#Более подробно о сути признаков можно прочитать по адресу 
#https://archive.ics.uci.edu/ml/datasets/Wine (см. также файл wine.names, приложенный к заданию)
data.columns = ['Class','Alcohol','MalicAcid','Ash','AlcalinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols','Proanthocyanins','ColorIntensity','Hue','OD280_OD315OfDilutedWines','Proline']
target = data.Class#вычленяем массив
model = data.drop(['Class'], axis=1)#Вычленяем модель из датасет 


#Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). 
#Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True). 
#Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42. 
#В качестве меры качества используйте долю верных ответов (accuracy).
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Вычислить качество на всех разбиениях можно при помощи функции sklearn.model_selection.cross_val_score. 
#В качестве параметра estimator передается классификатор, 
#в качестве параметра cv — генератор разбиений с предыдущего шага. 
#С помощью параметра scoring можно задавать меру качества, 
#по умолчанию в задачах классификации используется доля верных ответов (accuracy). 
#Результатом является массив, значения которого нужно усреднить.

#Найдите точность классификации на кросс-валидации для метода k ближайших соседей 
#(sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. 
#При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? 
#Данные результаты и будут ответами на вопросы 1 и 2.


def CalculateScores(model,count):        
    validationTest={}
    for k in range(50):#счетчик идет от нуля
        model_knc = KNeighborsClassifier(n_neighbors = (k+1)) #в параметре передаем кол-во соседей
        scores = cross_val_score(model_knc, model, target, scoring='accuracy',cv=kf)
        validationTest[k+1]=scores.mean()#берем среднее значение оценки

    #формируем датасет для сортировки    
    validationTestDataFrame=pandas.DataFrame.from_dict(validationTest, orient='index')#получаем из словаря датасет  
    validationTestDataFrame.index.name = 'k'
    validationTestDataFrame.columns =['Scores']
    validationTestDataFrame.sort_values(['Scores'], ascending=[False],inplace=True)
    print('При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? Данные результаты и будут ответами на вопросы 1 и 2.');
    print(validationTestDataFrame.head(count))

CalculateScores(model,1)
#Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. 
model2=scale(model)#масштабирование выполняется перед обучением
CalculateScores(model2,11)#Снова найдите оптимальное k на кросс-валидации.

#Какое значение k получилось оптимальным после приведения признаков к одному масштабу? 
#Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?




