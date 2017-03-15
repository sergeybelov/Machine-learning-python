# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:55:41 2017


"""
from skimage.io import imread
import pylab
from  skimage import img_as_float
import pandas as pd
from sklearn.cluster import KMeans
from math import log10
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
        imgMatrix[index]=imgFloat[x][y]
        index+=1

X_train=pd.DataFrame(imgMatrix).transpose()
X_train.columns=['R','G','B']
del imgMatrix

etalon=X_train.copy()
#==============================================================================
# Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241.
# После выделения кластеров все пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами:
# медианным и средним цветом по кластеру.
#==============================================================================
def psnr(X,X_train_true):#Расчет PSNR по цветному изображению в float числах
        mse=0
        for col in X_train_true.columns:
            mse+=((X[col]-X_train_true[col])**2).sum()
        mse/=len(X_train_true)*3
        return 10*log10(1/mse)#в целочисленном формате в числителе было бы 255, в float - 1


for clst in range(21):
    X_train=etalon.copy()
    maxClusters=clst+1

    cls=KMeans(init='k-means++',random_state=241,n_clusters=maxClusters)#обучаем кластеризатор
    kmeans = cls.fit(X_train)

    X_train['cluster']=kmeans.labels_#Добавляем классификацию кластера как колонку
    X_train.set_index('cluster', inplace=True)#Делаем новую колонку индексом
    X_train2=X_train.copy()#копируем данные
    X_train_true=X_train.copy()#копируем данные


    #замена цвета
    for cluster in range(maxClusters):
        for col in X_train.columns:
            median=X_train.loc[cluster,col].median()#считаем медиану интенсивности по кластеру
            X_train.loc[cluster,col]=median

            mean=X_train2.loc[cluster,col].mean()#считаем среднее интенсивности по кластеру
            X_train2.loc[cluster,col]=mean

    #==============================================================================
    # Измерьте качество получившейся сегментации с помощью метрики PSNR.
    # Эту метрику нужно реализовать самостоятельно (см. определение).
    # Найдите минимальное количество кластеров, при котором значение PSNR выше 20
    # (можно рассмотреть не более 20 кластеров, но не забудьте рассмотреть оба способа заполнения пикселей одного кластера).
    # Это число и будет ответом в данной задаче.
    #==============================================================================
    psnrMedian=psnr(X_train,X_train_true)
    psnrMean=psnr(X_train2,X_train_true)

    print("Clusters count %d, psnrMedian=%f, psnrMean=%f" % (maxClusters,psnrMedian,psnrMean))
    if (psnrMedian>=20) | (psnrMean>=20): break
