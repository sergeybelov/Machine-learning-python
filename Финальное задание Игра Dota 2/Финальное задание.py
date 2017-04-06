# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:40:37 2017


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
#from scipy.sparse import csr_matrix

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
# Описание признаков в таблице
# match_id: идентификатор матча в наборе данных
# start_time: время начала матча (unixtime)
# lobby_type: тип комнаты, в которой собираются игроки (расшифровка в dictionaries/lobbies.csv)
# Наборы признаков для каждого игрока (игроки команды Radiant — префикс rN, Dire — dN):
#   r1_hero: герой игрока (расшифровка в dictionaries/heroes.csv)
#   r1_level: максимальный достигнутый уровень героя (за первые 5 игровых минут)
#   r1_xp: максимальный полученный опыт
#   r1_gold: достигнутая ценность героя
#   r1_lh: число убитых юнитов
#   r1_kills: число убитых игроков
#   r1_deaths: число смертей героя
#   r1_items: число купленных предметов
# Признаки события "первая кровь" (first blood). Если событие "первая кровь" не успело произойти за первые 5 минут, то признаки принимают пропущенное значение
#   first_blood_time: игровое время первой крови
#   first_blood_team: команда, совершившая первую кровь (0 — Radiant, 1 — Dire)
#   first_blood_player1: игрок, причастный к событию
#   first_blood_player2: второй игрок, причастный к событию
# Признаки для каждой команды (префиксы radiant_ и dire_)
#   radiant_bottle_time: время первого приобретения командой предмета "bottle"
#   radiant_courier_time: время приобретения предмета "courier"
#   radiant_flying_courier_time: время приобретения предмета "flying_courier"
#   radiant_tpscroll_count: число предметов "tpscroll" за первые 5 минут
#   radiant_boots_count: число предметов "boots"
#   radiant_ward_observer_count: число предметов "ward_observer"
#   radiant_ward_sentry_count: число предметов "ward_sentry"
#   radiant_first_ward_time: время установки командой первого "наблюдателя", т.е. предмета, который позволяет видеть часть игрового поля
# Итог матча (данные поля отсутствуют в тестовой выборке, поскольку содержат информацию, выходящую за пределы первых 5 минут матча)
# duration: длительность
# radiant_win: 1, если победила команда Radiant, 0 — иначе
# Состояние башен и барраков к концу матча (см. описание полей набора данных)
#   radiant_win
#   tower_status_dire
#   barracks_status_radiant
#   barracks_status_dire
#==============================================================================

#==============================================================================
# Проверьте выборку на наличие пропусков с помощью функции count(),
# которая для каждого столбца показывает число заполненных значений.
# Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и
# попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.
#
#==============================================================================
train_size=len(data_train)
print("Select count=%s" % train_size)
for col in data_train.columns.values.tolist():
    count=data_train[col].count()
    if count!=train_size:
        print("Column %s, len=%s" % (col,count))

#==============================================================================
# Select count=97230
# Column first_blood_time, len=77677 - Если событие "первая кровь" не успело произойти за первые 5 минут, то признаки принимают пропущенное значение
# Column first_blood_team, len=77677 - Если событие "первая кровь" не успело произойти за первые 5 минут, то признаки принимают пропущенное значение
# Column first_blood_player1, len=77677
# Column first_blood_player2, len=53243
# Column radiant_bottle_time, len=81539
# Column radiant_courier_time, len=96538
# Column radiant_flying_courier_time, len=69751
# Column radiant_first_ward_time, len=95394
# Column dire_bottle_time, len=81087
# Column dire_courier_time, len=96554
# Column dire_flying_courier_time, len=71132
# Column dire_first_ward_time, len=95404
#==============================================================================
verbose=1
#==============================================================================
# Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным для логистической регрессии,
# поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание.
# Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение —
# в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева.
# Также есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании,
# но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.
#==============================================================================

# индекс, по которому будем отделять обучающую выборку от тестовой
idx_split = data_train.shape[0]#индекс последнего элемента
data_full = pd.concat([data_train, data_test])#формируем генеральную выборку из тренирующей и тестовой для одинаковой обработки
#del data_train, data_test#удаляем дата сеты чтобы не мешались в памяти

#Нужно найти все заполненные элементы
#Вычислить максимум

for col in data_full.columns.values.tolist():
    maxVal=data_full.loc[data_full[col].notnull(),col].max()**2#max()**3#Считаем максимум по всем заполненным значением и берем квадрат
    data_full.loc[data_full[col].isnull(),col]=maxVal#Заполняем все незаполненные значения данным результатом


#data_train.fillna(0, method=None, axis=1, inplace=True)


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

kf = KFold(n_splits=5,shuffle=True)#Конструктор кросс-валидации


#Проверяем гипотезу со стандартными настройками и заданным количеством деревьев
print('Проверяем гипотезу со стандартными настройками и заданным количеством деревьев')
for est in range(10,31,10):
    clf=GradientBoostingClassifier(n_estimators=est, random_state=241)#max_depth=3, n_estimators=70 Оценка качества=70.26 #**clf_grid.best_params_)#Передаем лучшие параметры в классификатор
    start_time = datetime.datetime.now()
    clf.fit(data_full.iloc[:idx_split, :], train_Y)#Обучаем
    print('Time elapsed:', datetime.datetime.now() - start_time)#замеряем время

    scores = cross_val_score(clf, data_full.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
    val=round(scores.mean()*100,2)#берем среднее значение оценки
    print("n_estimators=%s, Оценка качества=%s" % (est,val))


#Проверяем гипотезу что увеличение количества деревьев улучшает качество с уменьшением глубины дерева и ускоряет процесс
print('Проверяем гипотезу что увеличение количества деревьев улучшает качество с уменьшением глубины дерева и ускоряет процесс')
param_grid  = {'n_estimators':[60,70],'max_depth': range(3,5),'max_features': ["log2"]}#параметры сетки тестирования алгоритма
clf_grid = GridSearchCV(GradientBoostingClassifier(random_state=241), param_grid,cv=kf, n_jobs=1,verbose=verbose,scoring='roc_auc')
clf_grid.fit(data_full.iloc[:idx_split, :], train_Y)
print("best_params")
print(clf_grid.best_params_)
print("best_score")
print(clf_grid.best_score_)


#Пропущенное значение - очень большое число ^2
#best_params {'max_depth': 4, 'max_features': 'log2', 'n_estimators': 70}
#best_score 0.702832366129
#Пропущенное значение - очень большое число ^3
#best_params {'max_depth': 4, 'max_features': 'log2', 'n_estimators': 70}
#best_score 0.7031630032570212.3566666666666666666660.2315.2333333333333333333333333333332
#Пропущенное значение - медиана
#best_params {'max_depth': 4, 'max_features': 'log2', 'n_estimators': 70}
#best_score 0.702924164135
#Оценка качества=70.29

clf=GradientBoostingClassifier(**clf_grid.best_params_)#max_depth=3, n_estimators=70 Оценка качества=70.26 #**clf_grid.best_params_)#Передаем лучшие параметры в классификатор
start_time = datetime.datetime.now()
clf.fit(data_full.iloc[:idx_split, :], train_Y)#Обучаем
print('Time elapsed:', datetime.datetime.now() - start_time)#замеряем время

scores = cross_val_score(clf, data_full.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
val=round(scores.mean()*100,2)#берем среднее значение оценки
print("Оценка качества при увеличении числа деревьев=%s" % val)


#получаем список показателей которые сильнее всего влияют на предсказания
featureImportances=pd.DataFrame(data=clf.feature_importances_)
featureImportances.sort_values([0],ascending=False,inplace=True)
listCol=data_full.columns.values.tolist()


#Оценка качества=70.25
#1: d2_gold=8.43
#2: r2_gold=8.3
#3: d5_gold=8.01
#4: d1_gold=7.87
#5: r1_gold=7.81
#6: d4_gold=7.77
#7: r4_gold=7.61
#8: r5_gold=7.47
#9: r3_gold=7.28
#10: d3_gold=7.07
#11: first_blood_player1=2.37
#12: radiant_boots_count=2.23

#==============================================================================
#Сводный график значений
# import seaborn as sns
# cols = ['d2_gold',
# 'r2_gold',
# 'd5_gold',
# 'd1_gold',
# 'r1_gold',
# 'd4_gold',
# 'r4_gold',
# 'r5_gold',
# 'r3_gold',
# 'd3_gold']
# sns.pairplot(data_train[cols])
#
#==============================================================================

print('Характеристики которые вносят наибольший вклад в модель')
count=1
for i in featureImportances.index:
    if featureImportances.loc[i][0]<0.02: break
    print("%s: %s=%s" %(count,listCol[i],round(featureImportances.loc[i][0]*100,2)))
    count+=1



#feature_importances_ : array, shape = [n_features]
#The feature importances (the higher, the more important the feature).

    #start_time = datetime.datetime.now()

    #scores = cross_val_score(clf, data_train, train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
    #print('Time elapsed:', datetime.datetime.now() - start_time)#замеряем время

    #val=round(scores.mean()*100,2)#берем среднее значение оценки
    #print("n_estimators=%s, val=%s\%" %(n_est,val))


#==============================================================================
# Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? - около минуты
# Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC. ~70%
# Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Думаю нет, время обучения увеличивается сильно, а качество повышается слабо
# Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев? Уменьшить глубину деревьев и количество признаков, потеря точности небольшая (max_depth=4, max_features=sqrt, n_estimators=50, score=0.700039, total=   3.9s), а выигрыш во времени на порядок, использовать признаки имеющие максимальный вес при бустинге
#==============================================================================

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
data_full = pd.concat([data_train, data_test])#формируем генеральную выборку из тренирующей и тестовой для одинаковой обработки
data_full.index=range(0,len(data_full))#реиндексируем, иначе поиск по индексу работает некорректно
del data_train, data_test#удаляем дата сеты чтобы не мешались в памяти
data_full.fillna(0, method=None, axis=1, inplace=True)#для логистической регрессии зануляем пустые значения


param_grid  = {'C': np.logspace(-3, -1, 10)}#параметры сетки тестирования алгоритма - логарифмическая
#param_grid  = {'C': np.linspace(0.003, 0.008, 20)}#параметры сетки тестирования алгоритма - линейная

#Поиск лучшего коэффициента обучения
#print('Поиск лучшего коэффициента обучения')

def getScoreLogisticRegression(text,data_train,saveToFile=False):
    clf_grid = GridSearchCV(LogisticRegression(random_state=241,n_jobs=-1), param_grid,cv=kf, n_jobs=1,verbose=verbose,scoring='roc_auc')
    clf_grid.fit(data_full.iloc[:idx_split, :], train_Y)
    #print(u"best_params")
    #print(clf_grid.best_params_)
    #print(u"best_score")
    #print(clf_grid.best_score_)


    lr=LogisticRegression(n_jobs=-1,random_state=241,**clf_grid.best_params_)#Создаем логистрическую регрессию с лучшими параметрами
    lr.fit(data_train.iloc[:idx_split, :], train_Y)#Обучаем
    scores = cross_val_score(lr, data_train.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)#Оценка алгоритма
    val=round(scores.mean()*100,2)#берем среднее значение оценки
    print("Оценка качества GridSearchCV (%s)=%s" % (text,val))

    y_pred=pd.DataFrame(data=lr.predict_proba(data_train.iloc[idx_split:, :]))#прогнозируем
    y_pred.sort_values([0],inplace=True)#сортируем
    print(u'min=',y_pred.iloc[0,1],'; max=',y_pred.iloc[y_pred.shape[0]-1,1])#1 - класс означает, что Radiant победил

    if saveToFile:
        y_pred.sort_index(inplace=True)
        y_pred.to_csv('Radiant win predict',columns=[1],index_label=['match_id'],header=['prediction'])



getScoreLogisticRegression("without scaling",data_full)

#best_params whithot scaling
#{'C': 1.0000000000000001e-05}
#best_score whithot scaling
#0.557243278127

#попробуем шкалировать
data_full_norm=pd.DataFrame(data=StandardScaler().fit_transform(data_full))
getScoreLogisticRegression("with scaling",data_full_norm)

del data_full_norm
#best_params with scaling
#{'C': 0.0037275937203149379}
#best_score with scaling
#0.71587143058

#best_params with scaling
#{'C': 0.0071968567300115215}
#best_score with scaling
#0.716081358437

#best_params with scaling
#{'C': 0.0047142857142857143}
#best_score with scaling
#0.715872172211

#Какое качество получилось у логистической регрессии над всеми исходными признаками? - Без нормализации дает значительно худший результат (0.557243278127), после нормализации чуть лучший чем градиентный бустинг (0.71587143058)
#Как оно соотносится с качеством градиентного бустинга?
#Чем вы можете объяснить эту разницу? - Влияют сами данные, их нормализация и присутствие категориальных признаков
#Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом? - Работает быстрее

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

#best_params  drop categories, with scaling
#{'C': 0.0037275937203149379}
#best_score  drop categories, with scaling
#0.715900195147

#best_params  drop categories, with scaling
#{'C': 0.0055263157894736847}
#best_score  drop categories, with scaling
#0.715911754352

#best_params  drop categories, with scaling
#{'C': 0.0042813323987193957}
#best_score  drop categories, with scaling
#0.715990354582

#best_params  drop categories, with scaling
#{'C': 0.0051794746792312128}
#best_score  drop categories, with scaling
#0.716022599382

#==============================================================================
# Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? - Обучается чуть лучше, едва заметно
# Чем вы можете объяснить это изменение? - уменьшение размерности модели, как минимум не ухудшает прогнозируемость, линейные модели не очень хорошо работают с категориальными объектами в отличии от деревьев
#==============================================================================

#На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают,
#какие именно герои играли за каждую команду.
#Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, чем другие.
#Выясните из данных, сколько различных идентификаторов героев существует в данной игре
#(вам может пригодиться фукнция unique или value_counts).
cols.remove('lobby_type')#elfkztv из списка колонок лишнюю, чтобы остались только герои
iid=pd.Series(data_full[cols].values.flatten()).drop_duplicates()
N=iid.shape[0]
iid=pd.DataFrame(data=list(range(N)),index=iid.tolist())#переводим в обычный массив, чтобы индексация была чистая
iid.sort_index(inplace=True)
print(u'сколько различных идентификаторов героев существует в данной игре: ',N)

#сколько различных идентификаторов героев существует в данной игре:  108

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
#x_pick = pd.DataFrame(index=data_full_norm.index,columns=range(0,N))#Датафрейм для dummy-переменных


#for match_id in data_full.index:
#   row=data_full.ix[match_id,cols]#делаем слайс по строке и по нужным колонкам
#   rowPick=x_pick.ix[match_id]
#   for j, col in enumerate(row):
#       rowPick[iid.ix[col,0]] = 1 if j<5 else -1#классификатор героя одной или другой команды
#
#x_pick.fillna(0, method=None, axis=1, inplace=True)

x_pick_d = pd.get_dummies(data_full[cols[5:]].astype('str'))
x_pick_r = pd.get_dummies(data_full[cols[:5]].astype('str'))
x_pick_r *= -1
x_pick_d.columns=[col[1:] for col in list(x_pick_d.columns)]#убираем в колонках принадлежность к классам
x_pick_r.columns=[col[1:] for col in list(x_pick_r.columns)]#убираем в колонках принадлежность к классам

x_pick=x_pick_d+x_pick_r
del x_pick_d,x_pick_r


print('Завершили. Time elapsed:', datetime.datetime.now() - start_time)#замеряем время

total=data_full_norm.join(x_pick,rsuffix='_',how='inner')#pd.DataFrame(data=np.concatenate([x_pick,data_full_norm],axis=1))
del x_pick,data_full_norm
getScoreLogisticRegression("dummy coding",total,True)


#getScoreLogisticRegression("sparse matrix",data_full_norm_norm_sparse)
#del data_full_norm_norm_sparse

#best_params  sparse matrix
#{'C': 0.071968567300115208}
#best_score  sparse matrix
#0.751559380639

#best_params  sparse matrix
#{'C': 0.037275937203149381}
#best_score  sparse matrix
#0.751590669294

#best_params  sparse matrix
#{'C': 0.061054022965853265}
#best_score  sparse matrix
#0.751886025993

#==============================================================================
# Какое получилось качество при добавлении "мешка слов" по героям? - 0.751559380639
# Улучшилось ли оно по сравнению с предыдущим вариантом? - да, улучшилось
# Чем вы можете это объяснить? Переход из категориального кодирования на dummy кодирование - улучшает прогнозирование по модели на линейных методах
#==============================================================================

#==============================================================================
# Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
#==============================================================================

#==============================================================================
# Select count=97230
# Column first_blood_time, len=77677
# Column first_blood_team, len=77677
# Column first_blood_player1, len=77677
# Column first_blood_player2, len=53243
# Column radiant_bottle_time, len=81539
# Column radiant_courier_time, len=96538
# Column radiant_flying_courier_time, len=69751
# Column radiant_first_ward_time, len=95394
# Column dire_bottle_time, len=81087
# Column dire_courier_time, len=96554
# Column dire_flying_courier_time, len=71132
# Column dire_first_ward_time, len=95404
# Проверяем гипотезу со стандартными настройками и заданным количеством деревьев
# Time elapsed: 0:00:04.749582
# n_estimators=10, Оценка качества=66.74
# Time elapsed: 0:00:08.723383
# n_estimators=20, Оценка качества=68.36
# Time elapsed: 0:00:13.144788
# n_estimators=30, Оценка качества=69.05
# Проверяем гипотезу что увеличение количества деревьев улучшает качество с уменьшением глубины дерева и ускоряет процесс
# Fitting 10 folds for each of 4 candidates, totalling 40 fits
# [Parallel(n_jobs=1)]: Done  40 out of  40 | elapsed:  2.5min finished
# best_params
# {'max_depth': 4, 'max_features': 'log2', 'n_estimators': 70}
# best_score
# 0.703331166405
# Time elapsed: 0:00:04.665730
# Оценка качества при увеличении числа деревьев=70.34
# Характеристики которые вносят наибольший вклад в модель
# 1: d2_gold=5.47
# 2: d1_gold=5.1
# 3: r2_gold=5.05
# 4: r1_gold=4.8
# 5: d3_gold=4.54
# 6: r4_gold=4.33
# 7: d4_gold=4.32
# 8: r5_gold=3.87
# 9: d5_gold=3.86
# 10: r3_gold=2.72
# 11: d5_lh=2.65
# 12: r3_lh=2.35
# 13: r4_lh=2.34
# 14: d1_lh=2.12
# 15: radiant_boots_count=2.09
# --------------------------
# Fitting 10 folds for each of 10 candidates, totalling 100 fits
# [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   52.9s finished
# Оценка качества GridSearchCV (without scaling)=51.35
# min= 0.481385096984 ; max= 0.48164115994
# Fitting 10 folds for each of 10 candidates, totalling 100 fits
# [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   49.8s finished
# Оценка качества GridSearchCV (with scaling)=71.63
# min= 0.0168678616386 ; max= 0.989858087529
# Fitting 10 folds for each of 10 candidates, totalling 100 fits
# [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   49.2s finished
# Оценка качества GridSearchCV (drop categories, with scaling)=71.64
# min= 0.0172399825669 ; max= 0.989852747144
# сколько различных идентификаторов героев существует в данной игре:  108
# Старт dummy кодирования...
# Завершили. Time elapsed: 0:03:41.448749
# Fitting 10 folds for each of 10 candidates, totalling 100 fits
# [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   49.6s finished
# Оценка качества GridSearchCV (dummy coding)=74.68
# min= 0.00775904001868 ; max= 0.990233686565
#==============================================================================
