# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:58:56 2017

"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#==============================================================================
# Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше.
# Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
#==============================================================================
data_train = pd.read_csv('features.csv',index_col='match_id')
data_test = pd.read_csv('features_test.csv',index_col='match_id')

train_Y=data_train['radiant_win']#Целевая переменная 1, если победила команда Radiant, 0 — иначе
columns_train_difference=data_train.columns.difference(data_test.columns.values.tolist()).tolist()#Удалите признаки, которых нет в тестовой выборке - получаем различие в колонках
data_train.drop(columns_train_difference, axis=1, inplace=True)#удаляем внутри датасета

#==============================================================================
# Проверьте выборку на наличие пропусков с помощью функции count(),
# которая для каждого столбца показывает число заполненных значений.
# Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и
# попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.
#==============================================================================
train_size=data_train.shape[0]
print("Select count=%s" % train_size)
for col in data_train.columns.values.tolist():
    count=data_train[col].count()
    if count!=train_size:
        print("Column %s, len=%s" % (col,count))

#==============================================================================
# Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным для логистической регрессии,
# поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание.
# Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение —
# в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева.
# Также есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании,
# но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.
#==============================================================================
# индекс, по которому будем отделять обучающую выборку от тестовой
idx_split = train_size#индекс последнего элемента
data_full = pd.concat([data_train, data_test])#формируем генеральную выборку из тренирующей и тестовой для одинаковой обработки
del data_train,data_test

#если заполнять большим значением, то качество модели улучшается на 0,5%
#for col in data_full.columns.values.tolist():
    #maxVal=data_full.loc[data_full[col].notnull(),col].max()**2
    #data_full.loc[data_full[col].isnull(),col]=maxVal#Заполняем все незаполненные значения данным результатом


#Заполняем все незаполненные значения нулем
data_full.fillna(0, method=None, axis=1, inplace=True)#зануляем пустые значения

#==============================================================================
# Какой столбец содержит целевую переменную? Запишите его название.
#==============================================================================
print('Целевая переменная=radiant_win')

#==============================================================================
# Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на имеющейся матрице
# "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold),
# не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени,
# и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества.
# Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
# попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30).
# Долго ли настраивались классификаторы?
# Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего, продолжит расти при дальнейшем его увеличении?
#==============================================================================
verbose=1

kf = KFold(n_splits=5,shuffle=True)#Конструктор кросс-валидации

#Проверяем гипотезу со стандартными настройками и заданным количеством деревьев
print('Проверяем гипотезу со стандартными настройками и заданным количеством деревьев')
for est in range(10,31,10):
    clf=GradientBoostingClassifier(n_estimators=est, random_state=241)#max_depth=3, n_estimators=70 Оценка качества=70.26 #**clf_grid.best_params_)#Передаем лучшие параметры в классификатор
    start_time = datetime.datetime.now()
    scores = cross_val_score(clf, data_full.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
    print('Time elapsed:', datetime.datetime.now() - start_time)#замеряем время
    val=round(scores.mean()*100,2)#берем среднее значение оценки
    print("n_estimators=%s, Оценка качества=%s" % (est,val))


#Проверяем гипотезу что увеличение количества деревьев улучшает качество с уменьшением глубины дерева и ускоряет процесс
print('Проверяем гипотезу что увеличение количества деревьев улучшает качество с уменьшением глубины дерева и ускоряет процесс')
param_grid  = {'n_estimators':[60,70],'max_depth': range(3,5),'max_features': ["log2"]}#параметры сетки тестирования алгоритма
clf_grid = GridSearchCV(GradientBoostingClassifier(random_state=241), param_grid,cv=kf, n_jobs=1,verbose=verbose,scoring='roc_auc')
clf_grid.fit(data_full.iloc[:idx_split, :], train_Y)
print("best_params")
print(clf_grid.best_params_)

clf=GradientBoostingClassifier(random_state=241,**clf_grid.best_params_)

start_time = datetime.datetime.now()
scores = cross_val_score(clf, data_full.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
print('Time elapsed:', datetime.datetime.now() - start_time)#замеряем время
val=round(scores.mean()*100,2)#берем среднее значение оценки
print("Оценка качества=%s" % (val))

print('--------------------------')
#==============================================================================
# Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией)
# с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга.
# Подберите при этом лучший параметр регуляризации (C). Какое наилучшее качество у вас получилось?
# Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу?
# Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
# Важно: не забывайте, что линейные алгоритмы чувствительны к масштабу признаков!
# Может пригодиться sklearn.preprocessing.StandartScaler.
#==============================================================================
param_grid  = {'C': np.logspace(-3, -1, 10)}#параметры сетки тестирования алгоритма - логарифмическая

def getScoreLogisticRegression(text,data_train,saveToFile=False):
    clf_grid = GridSearchCV(LogisticRegression(random_state=241,n_jobs=-1), param_grid,cv=kf, n_jobs=1,verbose=verbose,scoring='roc_auc')
    clf_grid.fit(data_full.iloc[:idx_split, :], train_Y)

    lr=LogisticRegression(n_jobs=-1,random_state=241,**clf_grid.best_params_)#Создаем логистрическую регрессию с лучшими параметрами
    lr.fit(data_train.iloc[:idx_split, :], train_Y)#Обучаем
    start_time = datetime.datetime.now()
    scores = cross_val_score(lr, data_train.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
    print('Time elapsed:', datetime.datetime.now() - start_time)#замеряем время
    val=round(scores.mean()*100,2)#берем среднее значение оценки
    print("Оценка качества GridSearchCV (%s)=%s" % (text,val))

    y_pred=pd.DataFrame(data=lr.predict_proba(data_train.iloc[idx_split:, :]))#прогнозируем
    y_pred.sort_values([1],inplace=True)#сортируем по значениям
    print(u'min=',y_pred.iloc[0,1],'; max=',y_pred.iloc[y_pred.shape[0]-1,1])#1 - класс означает, что Radiant победил

    if saveToFile:
        y_pred.sort_index(inplace=True)
        y_pred.to_csv('Radiant win predict',columns=[1],index_label=['match_id'],header=['prediction'])

#обучаем без шкалирования
getScoreLogisticRegression("without scaling",data_full)

#попробуем шкалировать
data_full_norm=pd.DataFrame(data=StandardScaler().fit_transform(data_full))
getScoreLogisticRegression("with scaling",data_full_norm)
#==============================================================================
# Среди признаков в выборке есть категориальные, которые мы использовали как числовые,
# что вряд ли является хорошей идеей. Категориальных признаков в этой задаче одиннадцать:
# lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero.
# Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором
# лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?
#==============================================================================
cols = ['r%s_hero' % i for i in range(1, 6)]+['d%s_hero' % i for i in range(1, 6)]#колонки героев
cols.append('lobby_type')

#удаляем категориальные данные, нормируем и повторно ищем лучший коэффициент
data_full_norm=pd.DataFrame(data=StandardScaler().fit_transform(data_full.drop(cols, axis=1)))
getScoreLogisticRegression("drop categories, with scaling",data_full_norm)
print(u'кол-во колонок в матрице: ',len(data_full_norm.columns))
del data_full_norm

#==============================================================================
# #На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают,
# #какие именно герои играли за каждую команду.
# #Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, чем другие.
# #Выясните из данных, сколько различных идентификаторов героев существует в данной игре
# #(вам может пригодиться фукнция unique или value_counts).
#==============================================================================
cols.remove('lobby_type')#Убираем из списка колонок лишнюю, чтобы остались только герои
iid=pd.Series(data_full[cols].values.flatten()).drop_duplicates()
N=iid.shape[0]
iid=pd.DataFrame(data=list(range(N)),index=iid.tolist())#переводим в обычный массив, чтобы индексация была чистая
iid.sort_index(inplace=True)#хеш для героев
print(u'сколько различных идентификаторов героев существует в данной игре: ',N)

#==============================================================================
# Воспользуемся подходом "мешок слов" для кодирования информации о героях.
# Пусть всего в игре имеет N различных героев.
# Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице,
# если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire.
# Ниже вы можете найти код, который выполняет данной преобразование.
# Добавьте полученные признаки к числовым, которые вы использовали во втором пункте данного этапа.
#==============================================================================
# N — количество различных героев в выборке
print('Старт dummy кодирования...')
start_time = datetime.datetime.now()
x_pick = pd.DataFrame(index=data_full.index,columns=range(0,N))#Датафрейм для dummy-переменных


for match_id in data_full.index:
   row=data_full.ix[match_id,cols]#делаем слайс по строке и по нужным колонкам
   rowPick=x_pick.ix[match_id]
   for j, col in enumerate(row):
       rowPick[iid.ix[col,0]] = 1 if j<5 else -1#классификатор героя одной или другой команды

x_pick.fillna(0, method=None, axis=1, inplace=True)
print('Завершили. Time elapsed:', datetime.datetime.now() - start_time)#замеряем время

totalFeatures=data_full.join(x_pick,rsuffix='_',how='outer')#pd.DataFrame(data=np.concatenate([x_pick,data_full_norm],axis=1))
del x_pick,iid

cols.append('lobby_type')
#Все нормируем и удаляем лишнее
totalFeatures_norm=StandardScaler().fit_transform(totalFeatures.drop(cols, axis=1))
#==============================================================================
# Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации.
# Какое получилось качество? Улучшилось ли оно? Чем вы можете это объяснить?
# Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных
# моделей (лучшей с точки зрения AUC-ROC на кросс-валидации).
# Убедитесь, что предсказанные вероятности адекватные — находятся на отрезке [0, 1],
# не совпадают между собой (т.е. что модель не получилась константной).
#==============================================================================
getScoreLogisticRegression("dummy coding",pd.DataFrame(data=totalFeatures_norm),True)
