# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:35:24 2017

@author: нзнегз
"""
import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

#Какое количество мужчин и женщин ехало на корабле? 
print('Какое количество мужчин и женщин ехало на корабле')
print(data.Sex.value_counts())


#Какой части пассажиров удалось выжить? 
#Посчитайте долю выживших пассажиров. 
#Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
print('Доля выживших пассажиров')
#mask = data['Survived'] == 1
#newdata=data[mask]
#print(data.groupby('Survived').count())
                                        
onePercentTotal=data.Survived.count()/100
survived=data.Survived.sum()
print(round(survived/onePercentTotal,2))

#Какую долю пассажиры первого класса составляли среди всех пассажиров? 
#Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
print('Доля пассажиров первого класса')
data['FirstClassCounter']= 0#добавляем колонку со значением по умолчанию
mask=data.Pclass == 1#маска выборки
data.loc[mask,'FirstClassCounter']=1#выборка по колонке - будет счетчиком
    
_firstClassCounter=data.FirstClassCounter.sum()
print(round(_firstClassCounter/onePercentTotal,2))


#Какого возраста были пассажиры? 
#Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два числа через пробел.
print('Посчитайте среднее и медиану возраста пассажиров')

print(round(data.Age.mean(),2))
print(round(data.Age.median(),2))

#Диаграмма
data['count']= 1
data.pivot_table('Name', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)

#Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? 
#Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
print('Коррелируют ли число братьев/сестер/супругов с числом родителей/детей')
df = pandas.DataFrame(data=data,columns=['SibSp', 'Parch'])
print(round(df.corr().loc['SibSp','Parch'],2))#Пересечение двух областей

#Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) 
#его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных. 
#Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. 
#Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен, 
#а также разделения их на женские и мужские.
print("Какое самое популярное женское имя на корабле")

NameExtract=data.Name.str.extract('^.+?Mrs\..+?\((.+?)\)|^.+?Miss\.\s([\w\s]+)',expand=False)#первичное преобразование
NameExtract.fillna('',inplace=True)#заменяем NULL на пустую строку
NameExtract.columns = ['Mrs','Miss']#переименуем колонки
NameExtract['CombinedName']=NameExtract.apply(lambda x:'%s%s' % (x['Mrs'],x['Miss']),axis=1)#соединяем в одну колонку
NameExtract.drop(['Mrs','Miss'],axis=1,inplace=True)#удаляем ненужные колонки
#print(NameExtract.CombinedName.str.split(' '))
    
femalenames=[]
#обрабатываем список имен
for i in range(NameExtract.CombinedName.count()):
    names=NameExtract.loc[i+1,'CombinedName']
    words=names.split(' ')
    for word in words:
        if len(word)>2 and word.find('"'):
            femalenames.append(word)
    
femalenamesDataFrame=pandas.DataFrame(data=femalenames,columns=['Names'])#получаем из листа датасет
femalenamesDataFrame['count']= 1#колонка счетчик
femalenamesDataFrameGroupBy=femalenamesDataFrame.groupby('Names').sum()#группируем по имени и суммируем счетчик
femalenamesDataFrameSorted=femalenamesDataFrameGroupBy.sort_values(['count'], ascending=[False])#сортируем по счетчику в порядке убывания
print(femalenamesDataFrameSorted.head(2))#показываем часто встечаемое имя



    
    