# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:55:41 2017


"""
from skimage.io import imread
import pylab
from  skimage import img_as_float
import pandas as pd
from sklearn.cluster import KMeans
#==============================================================================
# Для работы с изображениями мы рекомендуем воспользоваться пакетом scikit-image.
# Чтобы загрузить изображение, необходимо выполнить следующую команду
#==============================================================================
image = imread('parrots.jpg')
#pylab.imshow(image)#Показ картинки

#==============================================================================
# Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1.
# Для этого можно воспользоваться функцией img_as_float из модуля skimage. Обратите внимание на этот шаг,
# так как при работе с исходным изображением вы получите некорректный результат.
#==============================================================================
imgFloat=img_as_float(image)
del image

#Создайте матрицу объекты-признаки: характеризуйте каждый пиксель
#тремя координатами - значениями интенсивности в пространстве RGB.
imgMatrix={}

index=0
for x in range(len(imgFloat)):
    for y in range(len(imgFloat[x])):
        I=imgFloat[x][y].sum()/3.#преобразовываем значениями интенсивности в пространстве RGB
        imgMatrix[index]={'x':x,'y':y,'I':I}
        index+=1

X_train=pd.DataFrame(imgMatrix).transpose()
del imgMatrix

#==============================================================================
# Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241.
# После выделения кластеров все пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами:
# медианным и средним цветом по кластеру.
#==============================================================================
cls=KMeans(init='k-means++',random_state=241)
kmeans = cls.fit(X_train)

X_train['cluster']=kmeans.labels_#Добавляем классификацию кластера как колонку
maxClusters=X_train.cluster.max()#максимальное разбиение кластеров

X_train.set_index('cluster', inplace=True)#Делаем новую колонку индексом
X_train2=X_train.copy()#копируем данные

for cluster in range(maxClusters):
    thisCluster=cluster+1
    median=X_train.loc[thisCluster,'I'].median()#считаем медиану интенсивности по кластеру
    X_train.loc[thisCluster,'I']=median

    mean=X_train2.loc[thisCluster,'I'].mean()#считаем среднее интенсивности по кластеру
    X_train2.loc[thisCluster,'I']=mean

