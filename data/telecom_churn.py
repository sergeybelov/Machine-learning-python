# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:09:39 2017


"""

# импортируем Pandas и Numpy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#открываем файл с данными
df = pd.read_csv('telecom_churn.csv')

print('Размер датафрейма')
print(df.shape)

print('Колонки')
print(df.columns)

print('Общая информация')
print(df.info())

#изменяем с булева на целый числовой
df['Churn'] = df['Churn'].astype('int64')

print('Основные статистические характеристики данных по каждому числовому признаку (типы int64 и float64)')
print(df.describe())

#Чтобы посмотреть статистику по нечисловым признакам, нужно явно указать интересующие нас типы в параметре include.
print('Статистика по нечисловым признакам')
print(df.describe(include=['object', 'bool']))

print('Распределение значений целевой переменной')
print(df['Churn'].value_counts())


#Посмотрим на распределение пользователей по переменной Area code.
#Укажем значение параметра normalize=True, чтобы посмотреть не абсолютные частоты, а относительные.
#Area code-Префикс номера телефона
print('распределение по префиксу номеров телефона')
print(df['Area code'].value_counts(normalize=True))#значения в процентах по частоте

print('Сортировка по группе столбцов')
print(df.sort_values(by=['Churn', 'Total day charge'],
        ascending=[True, False]).head())

#логическая выборка
print('Cколько в среднем в течение дня разговаривают по телефону нелояльные пользователи?')
print(df[df['Churn'] == 1]['Total day minutes'].mean())

print('Какова максимальная длина международных звонков среди лояльных пользователей (Churn == 0), не пользующихся услугой международного роуминга ("International plan" == "No")?')
print(df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max())

print('Только первая строка')
print(df[:1])

print('Только последняя строка')
print(df[-1:])


print('используем map для обработки каждой строки')
d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
print(df.head())


#==============================================================================
# Группировка данных
# df.groupby(by=grouping_columns)[columns_to_show].function()
# К датафрейму применяется метод groupby, который разделяет данные по grouping_columns – признаку или набору признаков.
# Выбираем нужные нам столбцы (columns_to_show).
# К полученным группам применяется функция или несколько функций.
#==============================================================================

#Группирование данных в зависимости от значения признака Churn и вывод статистик по трём столбцам в каждой группе.

columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']
print(df.groupby(['Churn'])[columns_to_show].describe(percentiles=[]))

#Сделаем то же самое, но немного по-другому, передав в agg список функций:
columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']
print(df.groupby(['Churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max]))


#==============================================================================
#Сводные таблицы
# Допустим, мы хотим посмотреть, как наблюдения в нашей выборке распределены в контексте двух признаков — Churn и International plan.
# Для этого мы можем построить таблицу сопряженности, воспользовавшись методом crosstab:
#==============================================================================
print('Кросс таблицы в абсолютном выражении')
print(pd.crosstab(df['Churn'], df['International plan']))

print('Кhосс таблицы, по процентам')
print(pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True))

#В Pandas за сводные таблицы отвечает метод pivot_table, который принимает в качестве параметров:
#values – список переменных, по которым требуется рассчитать нужные статистики,
#index – список переменных, по которым нужно сгруппировать данные,
#aggfunc — то, что нам, собственно, нужно посчитать по группам — сумму, среднее, максимум, минимум или что-то ещё.

#Area code - строки
#'Total day calls', 'Total eve calls', 'Total night calls' - столбцы
#aggfunc='mean' - аггрегирующая функция в ячейках
print('Сводные таблицы')
print(df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'], ['Area code'], aggfunc='mean').head(10))

#добавление столбца как суммы других столбцов
df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']

#drop - удаление столбца и строк

print('--- Первые попытки прогнозирования оттока ---')
print('Посмотрим, как отток связан с признаком "Подключение международного роуминга" (International plan). Сделаем это с помощью сводной таблички crosstab')
#margins=True - общий итог
print(pd.crosstab(df['Churn'], df['International plan'], margins=True))

print('Далее посмотрим на еще один важный признак – "Число обращений в сервисный центр" (Customer service calls). Также построим сводную таблицу и картинку.')
print(pd.crosstab(df['Churn'], df['Customer service calls'], margins=True))

#Добавим колонку количество звонков в техподдержку
df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')#по сути булево значение
print(pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True))

print('Объединенная статистика')
print(pd.crosstab(df['Many_service_calls'] & df['International plan'] , df['Churn'], normalize=True))

#Доля лояльных клиентов в выборке – 85.5%. Самая наивная модель, ответ которой "клиент всегда лоялен"
#на подобных данных будет угадывать примерно в 85.5% случаев.
#То есть доли правильных ответов (accuracy) последующих моделей должны быть как минимум не меньше, а лучше, значительно выше этой цифры;
print('Доля лояльных клиентов в выборке')
print(round(df['Churn'].value_counts(normalize=True)[0]*100,2))

#С помощью простого прогноза, который условно можно выразить такой формулой:
#"International plan = False & Customer Service calls < 4 => Churn = 0, else Churn = 1",
#можно ожидать долю угадываний 85.8%, что еще чуть выше 85.5%.
#Впоследствии мы поговорим о деревьях решений и разберемся, как находить подобные правила автоматически на основе только
#входных данных;

#Перед обучением сложных моделей рекомендуется немного покрутить данные и
#проверить простые предположения. Более того, в бизнес-приложениях машинного обучения чаще
#всего начинают именно с простых решений, а потом экспериментируют с их усложнениями.


#Посмотрим на распределение целевого класса – оттока клиентов.
#plt=df['Churn'].value_counts().plot(kind='bar', label='Churn',title='Распределение оттока клиентов',legend = True)

#Выделим следующие группы признаков (среди всех кроме Churn ):
#1. бинарные: International plan, Voice mail plan
#2. категориальные: State
#3. порядковые: Customer service calls
#5. количественные: все остальные
#
#Посмотрим на корреляции количественных признаков. По раскрашенной матрице корреляций видно,
#что такие признаки как Total day charge считаются по проговоренным минутам (Total day minutes).
#То есть 4 признака можно выкинуть, они не несут полезной информации.

corr_matrix = df.drop(['State', 'International plan', 'Voice mail plan',
                      'Area code'], axis=1).corr()
#sns.heatmap(corr_matrix);
#print(corr_matrix.index)

#линейно зависимые признаки
#Total day minutes - Total day charge
#Total eve minutes - Total eve charge
#Total night minutes - Total night charge
#Total intl minutes - Total intl charge


#Теперь посмотрим на распределения всех интересующих нас количественных признаков.
#выкидываем ненужные колонки
features = df.drop(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge', 'Churn'],axis=1)

#features.hist(figsize=(20,12))


#Видим, что большинство признаков распределены нормально.
#Исключения – число звонков в сервисный центр (Customer service calls)
#(тут больше подходит пуассоновское распределение) и
#число голосовых сообщений (Number vmail messages, пик в нуле, т.е. это те,
#у кого голосовая почта не подключена). Также смещено распределение числа международных звонков (Total intl calls).

#Еще полезно строить вот такие картинки, где на главной диагонали рисуются распредления признаков,
#а вне главной диагонали – диаграммы рассеяния для пар признаков. Бывает, что это приводит к каким-то выводам,
#но в данном случае все примерно понятно, без сюрпризов.
featuresList = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge','Churn']))

#df[features].hist(figsize=(20,12))
#sns.pairplot(df[featuresList + ['Churn']], hue='Churn')

#Построим boxplot-ы, описывающее статистики распределения количественных признаков в двух группах:
#среди лояльных и ушедших клиентов.

#fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

#for idx, feat in  enumerate(features):
    #sns.boxplot(x='Churn', y=feat, data=df, ax=axes[idx / 4, idx % 4])
    #axes[idx / 4, idx % 4].legend()
    #axes[idx / 4, idx % 4].set_xlabel('Churn')
    #axes[idx / 4, idx % 4].set_ylabel(feat);


#На глаз наибольшее отличие мы видим для признаков Total day minutes, Customer service calls и Number vmail messages.

#_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

#sns.boxplot(x='Churn', y='Total day minutes', data=df, ax=axes[0]);
#sns.violinplot(x='Churn', y='Total day minutes', data=df, ax=axes[1]);

#Интересное наблюдение: в среднем ушедшие клиенты больше пользуются связью.
#Возможно, они недовольны тарифами, и одной из мер борьбы с оттоком будет понижение тарифных ставок
#(стоимости мобильной связи).
#Но это уже компании надо будет проводить дополнительный экономический анализ,
#действительно ли такие меры будут оправданы.

#Теперь изобразим распределение числа обращений в сервисный центр.
#Тут уникальных значений признака не много (признак можно считать как количественным целочисленным,
#так и порядковым), и наглядней изобразить распределение с помощью countplot.
#Наблюдение: доля оттока сильно возрастает начиная с 4 звонков в сервисный центр.

#sns.countplot(x='Customer service calls', hue='Churn', data=df)

#Теперь посмотрим на связь бинарных признаков International plan и Voice mail plan с оттоком.
#Наблюдение: когда роуминг подключен, доля оттока намного выше,
#т.е. наличие международного роуминга – сильный признак. Про голосовую почту такого нельзя сказать.

#_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

#sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0]);
#sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1]);

#Наконец, посмотрим, как с оттоком связан категориальный признак State.
#С ним уже не так приятно работать, поскольку число уникальных штатов довольно велико – 51.
#Можно в начале построить сводную табличку или посчитать процент оттока для каждого штата.
#Но данных по каждом штату по отдельности маловато (ушедших клиентов всего от 3 до 17 в каждом штате),
#поэтому, возможно, признак State впоследствии не стоит добавлять в модели классификации из-за риска
# переобучения (но мы это будем проверять на кросс-валидации, stay tuned!).
#Доли оттока для каждого штата:

print('Штаты с большим оттоком')
StateChurn=df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False)
print(StateChurn.head(10))

# преобразуем все признаки в числовые, выкинув штаты
X = df.drop(['Churn', 'State'], axis=1)
X['International plan'] = pd.factorize(X['International plan'])[0]
X['Voice mail plan'] = pd.factorize(X['Voice mail plan'])[0]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


tsne = TSNE(random_state=17)
tsne_representation = tsne.fit_transform(X_scaled)

#plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
            #c=df['Churn'].map({0: 'green', 1: 'red'}));

_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

axes[0].scatter(tsne_representation[:, 0], tsne_representation[:, 1],
            c=df['International plan'].map({'Yes': 'green', 'No': 'red'}))
axes[1].scatter(tsne_representation[:, 0], tsne_representation[:, 1],
            c=df['Voice mail plan'].map({'Yes': 'green', 'No': 'red'}))

axes[0].set_title('International plan')
axes[1].set_title('Voice mail plan')