# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:43:34 2017


"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('howpop_train.csv')

#Избавимся сразу от переменных, названия которых заканчиваются на _lognorm (нужны для соревнования на Kaggle). Выберем их с помощью filter() и удалим drop-ом:
df.drop(filter(lambda c: c.endswith('_lognorm'), df.columns),
        axis = 1,       # axis = 1: столбцы
        inplace = True) # избавляет от необходимости сохранять датасет

# настройка внешнего вида графиков в seaborn
sns.set_style("dark")
sns.set_palette("RdBu")
sns.set_context("notebook", font_scale = 1.5,
                rc = { "figure.figsize" : (25, 5), "axes.titlesize" : 18 })


#Столбец published (время публикации) содержит строки. Чтобы мы могли работать с этими данными как с
#датой/временем публикации, приведём их к типу datetime:
#print(df.published.dtype)
df['published'] = pd.to_datetime(df.published, yearfirst = True)
#print(df.published.dtype)


#Создадим несколько столбцов на основе данных о времени публикации:
df['year'] = [d.year for d in df.published]
df['month'] = [d.month for d in df.published]

df['dayofweek'] = [d.isoweekday() for d in df.published]
df['hour'] = [d.hour for d in df.published]
df['dayofmonth'] = [d.day for d in df.published]


#В каком месяце (и какого года) было больше всего публикаций?
#dateFrame=df[(df.year>2014) & (df.month==3)].groupby(['year','month'],sort=True)['post_id'].count()

dateFrame=df[df.year>2014].groupby(['year','month'],sort=True)['post_id'].count()
#dateFrame.plot(kind='bar', rot=90,grid=True)
dateFrame.sort_values(ascending=False, inplace=True)

val=dateFrame.head(1).index.values[0]#численный максимум - берем имя индекса
#print('Максимум на %d.%d' %(val[0],val[1]))

del dateFrame
#март 2015

#Проанализируйте публикации в этом месяце (из вопроса 1), отметьте все подходящие утверждения
dateFrame=df[(df.year==val[0])&(df.month==val[1])].copy()#делаем выборку за указанный период, копированием датасета
#dateFrame=dateFrame.groupby('dayofweek')['post_id'].count().T
#dateFrame.plot(kind='bar', rot=90,grid=True)

#в субботу и воскресенье проседание - публикуют меньше всего

#На хабре всегда больше статей, чем на гиктаймсе
dateFrame['habr']=dateFrame.domain.map(lambda x: 1 if x.startswith('habrahabr.ru') else 0)
dateFrame['geek']=dateFrame.domain.map(lambda x: 1 if x.startswith('geektimes.ru') else 0)

#habrGeek=dateFrame.groupby(['dayofmonth'],sort=True)['habr','geek'].sum()
#habrGeek.plot(kind='bar', rot=90,grid=True)
#ответ нет, дни 14, 21


#По субботам на гиктаймс и на хабрахабр публикуют примерно одинаковое число статей
habrGeek=dateFrame[dateFrame.dayofweek==6].groupby(['dayofweek'],sort=True)['habr','geek'].sum()
#habrGeek.plot(kind='bar', rot=90,grid=True)
#ответ Да
del dateFrame
del habrGeek

#Больше всего просмотров набирают статьи, опубликованные в 12 часов дня?
dateFrame=df.groupby(['hour'],sort=True)['views'].sum()
dateFrame.sort_values(ascending=False, inplace=True)
#dateFrame.plot(kind='bar', rot=90,grid=True, title='Просмотры')
#оттвет ДА
del dateFrame

#У опубликованных в 10 утра постов больше всего комментариев
dateFrame=df.groupby(['hour'],sort=True)['comments'].sum()
#dateFrame.plot(kind='bar', rot=90,grid=True, title='Комментарии')
#ответ НЕТ, правильно в 13 часов
del dateFrame

#Больше всего просмотров набирают статьи, опубликованные в 6 часов утра
#Ответ - НЕТ

#Максимальное число комментариев на гиктаймсе набрала статья, опубликованная в 9 часов вечера
dateFrame=df.copy()
dateFrame['geek']=dateFrame.domain.map(lambda x: x.startswith('geektimes.ru'))

select=dateFrame[dateFrame.geek==True].groupby(['hour'],sort=True)['comments'].sum()
#select.plot(kind='bar', rot=90,grid=True, title='Комментарии к geektimes')
#Ответ - НЕТ,пhавильный ответ 12 часов
del dateFrame
del select


#На хабре дневные статьи комментируют чаще, чем вечерние
dateFrame=df.copy()
dateFrame['habr']=dateFrame.domain.map(lambda x: x.startswith('habrahabr.ru'))
dateFrame['day_comments']=(dateFrame[['comments','hour']]).apply(lambda x: x.comments if (x.hour>=12)and(x.hour<18) else 0,axis =1)
dateFrame['eve_comments']=(dateFrame[['comments','hour']]).apply(lambda x: x.comments if (x.hour>=18)and(x.hour<=23) else 0,axis =1)

dateFrame=dateFrame[(dateFrame.habr==True)&(dateFrame.hour>=12)].groupby(['hour'],sort=True)['day_comments','eve_comments'].sum()
dateFrame.plot(kind='bar', rot=45,grid=True,title='На хабре дневные статьи комментируют чаще, чем вечерние')
#Ответ ДА

#Кого из топ-20 авторов (по числу статей) чаще всего минусуют (в среднем)
dateFrame=df.copy()
dateFrame=dateFrame.groupby(['author'],sort=True)['post_id'].count()
dateFrame.sort_values(ascending=False, inplace=True)
topAuthor=dateFrame.head(20).index.values

dateFrame=df[df.author.isin(topAuthor)].groupby(['author'],sort=True)['votes_minus'].count()
dateFrame.plot(kind='bar',grid=True, rot=45)
#Ответ @alizar

#Сравните субботы и понедельники. Правда ли, что по субботам авторы пишут в основном днём,
#а по понедельникам — в основном вечером?
dateFrame=df.copy()
dateFrame=dateFrame[(dateFrame.dayofweek.isin([1,6])) & (dateFrame.hour>=12)].groupby(['dayofweek','hour'],sort=True)['post_id'].count()
dateFrame.plot(kind='bar',grid=True, rot=45)