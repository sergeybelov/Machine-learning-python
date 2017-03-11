# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:09:39 2017


"""

# импортируем Pandas и Numpy
import pandas as pd
import numpy as np

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