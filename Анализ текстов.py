# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:08:44 2017

@author: tehn-11
"""

#==============================================================================
#  Анализ текстов
#==============================================================================
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

#==============================================================================
# Для начала вам потребуется загрузить данные.
# В этом задании мы воспользуемся одним из датасетов, доступных в scikit-learn'е — 20 newsgroups.
# Для этого нужно воспользоваться модулем datasets:
#==============================================================================
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

#==============================================================================
# Вычислите TF-IDF-признаки для всех текстов.
# Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным.
# При таком подходе получается, что признаки на обучающем множестве используют информацию из тестовой выборки — но такая ситуация вполне законна,
# поскольку мы не используем значения целевой переменной из теста.
# На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения,
# и поэтому можно ими пользоваться при обучении алгоритма.
#==============================================================================

#В Scikit-Learn это реализовано в классе
#sklearn.feature_extraction.text.TfidfVectorizer.
#Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.
y_train = newsgroups.target#Класс
X_train = newsgroups.data#Характеристики

vectorizer = TfidfVectorizer()
dataMatrix=vectorizer.fit_transform(X_train).toarray()#матрица объектов по словам, в ячейках веса слов

#idf = vectorizer.idf_
words=vectorizer.get_feature_names()
#tf_idf=dict(zip(words, idf))#токены с весами


#==============================================================================
# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear')
# при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для KFold.
# В качестве меры качества используйте долю верных ответов (accuracy).
#==============================================================================

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(dataMatrix, y_train)#внимание очень долго работает на тестовых данных

#записываем параметры в массив
#validationTest=dict(zip(gs.cv_results_['mean_test_score'], gs.cv_results_['params']))#получаем инфу по лучшему параметру в виде массива, который склеиваем с оценкой качества
#validationTestData=pd.DataFrame(data=validationTest).transpose()#создаем датафрейм с транспонированием, покольку значения полявляются в колонках, а не строках
#validationTestData.sort_index(ascending=[False],inplace=True)

#У GridSearchCV есть поле best_estimator_,
#которое можно использовать, чтобы не обучать заново классификатор с оптимальным параметром.
bestC=gs.best_estimator_.C

#устарело
#for a in gs.grid_scores_:
 #   validationTest[a.mean_validation_score]=a.parameters# — оценка качества по кросс-валидации и значения параметров

#==============================================================================
# Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
#==============================================================================
clf = SVC(C=bestC,kernel='linear', random_state=241)
result=clf.fit(dataMatrix, y_train)


#==============================================================================
# Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC).
# Они являются ответом на это задание. Укажите эти слова через запятую или пробел, в нижнем регистре,
#в лексикографическом порядке.
#==============================================================================
weights=[]
for element in result.coef_.T:#с транспонированием, иначе все коэффициенты идут строкой
    weights.append(abs(element[0]))

combine=dict(zip(words,weights))
bestWords=pd.DataFrame(data=combine,index=[0]).transpose()#с транспонированием, иначе все коэффициенты идут строкой
bestWords.columns = ['weights']#переименуем колонки
bestWords.sort_values(['weights'], ascending=[False],inplace=True)

topTenWordsCollection=bestWords.head(10).index.values#берем значения 10 индексов (там слова)
newWords=pd.DataFrame(data=topTenWordsCollection)
newWords.columns = ['words']#переименуем колонки
newWords.words.astype(str)
newWords=newWords.apply(lambda x: x[0].lower(),axis =1)#приводим к нижнему регистру, имя колонки теряется
newWords.sort_values(0, ascending=[True],inplace=True)#сортируем в лексикографическом порядке.

collection=newWords.values#получаем значения массива
print (' '.join(map(lambda x: x, collection)))#сливаем в строку каждый элемент

