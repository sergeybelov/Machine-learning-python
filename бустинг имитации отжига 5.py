# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:19:37 2019

@author: Belov
"""
#LICENSE ON christofides_tsp
#=======
#Copyright (c) 2016 D. S. Rahul
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Имитация отжига (С) 2019,2016 Белов С.В.,D. S. Rahul
#mail@belovsergey.ru

import pandas as pd
import os
import numpy as np
from itertools import cycle, dropwhile, islice
import random
from multiprocessing import Process, Queue
import tqdm
import sys
import re
import io
import json
from pytsp.christofides_tsp import christofides_tsp
import time
#----------------------------------------------


def read_file(file):
    file_data=pd.read_csv(file , sep=';', header=0,encoding='utf-8')#читаем файл

    #Трансформируем в матрицу
    #Сначала получаем список уникальных ячеек
    cols=np.concatenate((file_data['Ячейка1'].unique(),file_data['Ячейка2'].unique()), axis=0)
    cols=np.unique(cols, axis=0)

    #теперь формируем матрицу значений с пустыми значениями
    data=pd.DataFrame(columns=cols,index=cols)

    #теперь заполняем матрицу, через итерацию по строкам
    for _, row in file_data.iterrows():
        weight=row.Расстояние
        rw=row.Ячейка1
        cl=row.Ячейка2
        data[rw][cl]=weight
        data[cl][rw]=weight
    return data.fillna(0).astype(float).values,cols



def route_cost(graph,result):
    first_id=result[0]
    cost=graph[result[-1]][first_id]
    for next_id in result[1:-1]:
        cost+=graph[first_id][next_id]
        first_id=next_id
    return cost



def simulated_annealing(graph, min_cost, path, alpha,queue):
    random.seed()
    current_route_cost=min_cost
    lp=len(path)


    #temperature=1.0#начальная температура
    #while temperature>1e-15:#Пока температура не опустится до нуля
    for temperature in np.logspace(0,10,num=lp**2,base=1e-1):#температура от 1 до 10-15 степени
        new_solution = path.copy()
        left_index = random.randint(2,  lp- 1)
        right_index = random.randint(0, lp - left_index)
        last=right_index + left_index
        new_solution[right_index: last] = reversed(new_solution[right_index: last])
        new_route_cost=route_cost(graph, new_solution)

        if new_route_cost < current_route_cost or np.exp((current_route_cost - new_route_cost) / temperature) > random.random():
            cycled = cycle(new_solution)
            skipped = dropwhile(lambda x: x != 0, cycled)
            sliced = islice(skipped, None, lp)
            path = list(sliced)
            current_route_cost =route_cost(graph,path)

        #temperature*=alpha

    queue.put([path,current_route_cost])


def _start_new_proc(processes,iter_limit,graph, cost, best_path, alpha, queue):
    while(len(processes)<iter_limit):
        p=Process(target=simulated_annealing, args=(graph, cost, best_path, alpha, queue,))
        p.start()
        processes.append([p,cost])

    return processes


#----------------------------
if __name__ == "__main__":
    if len (sys.argv) != 4:
        raise ValueError("Отсутствуют ключи <адрес файла для расчета> <адрес файла для результата>  -iter_limit<Количество одновременных процессов (больше - лучше, но дольше и идет нагрузка на процессор)>")

    rootDir = os.path.abspath(os.curdir)
    file=rootDir+'\\'+sys.argv[1]
    file_result=rootDir+'\\'+sys.argv[2]

    print('in: '+file)
    print('out: '+file_result)

    match = re.search(r'-iter_limit(\d+)', sys.argv[3])
    iter_limit=int(match[1])


    graph,columns=read_file(file)

    #начальный путь
    initial_path = christofides_tsp(graph)
    initial_min_cost=route_cost(graph, initial_path)
    print('initial cost: '+str(initial_min_cost))

    queue = Queue()

    size=len(initial_path)#размер пути
    alpha=1-1e-3#параметр уменьшения температуры
    iteration=size**2#количество итераций изменений
    total_lim=size#глобальный лимит итераций улучшения

    if not iter_limit:
        raise ValueError("Нулевой iter_limit")

    limit=iter_limit#минимальное количество не улучшений, чтобы выйти из основного цикла
    if limit<5: limit=5
    print('iter_limit: '+str(iter_limit))

    pbar = tqdm.tqdm(desc='calculate',mininterval=2, maxinterval=5)

    best_path = initial_path.copy()
    min_cost=initial_min_cost


    #создаем процессы
    match_eqv=limit
    processes=_start_new_proc([],iter_limit,graph, initial_min_cost, initial_path, alpha, queue)
    while(processes):
        #проверяем какие процессы закончились
        for ind,el in enumerate(processes):
            p=el[0]
            if p.is_alive(): continue

            recent_cost=el[1]
            p.close()
            processes.pop(ind)
            total_lim-=1


            #выбераем лучший путь
            if not queue.empty():
                path,cost = queue.get()
                if cost<min_cost:#глобальный результат
                    min_cost,best_path=cost,path

                if cost<recent_cost:#локальный результат
                    match_eqv=limit
                else:
                    cost,path=min_cost,best_path
                    match_eqv-=1

                pbar.set_description(desc='iteration left: '+str(match_eqv)+', min_cost: '+str(min_cost)+', recent_cost: '+str(recent_cost)+', curr_cost: '+str(cost), refresh=False)
                pbar.update(1)
                if match_eqv>=0 and total_lim>=0: #новые процессы не создаем - вышли за лимиты итераций улучшения
                    #запускаем новый процесс
                    processes=_start_new_proc(processes,len(processes)+1,graph, cost, path, alpha, queue)
            break

        else:
            time.sleep(0.03)


    pbar.set_description(desc='Done.', refresh=True)
    pbar.close()
    print('final cost: '+str(min_cost))


    #эмпирическая оценка улучшения маршрута
    upgrade_value=float(min_cost)/initial_min_cost
    print('upgrade_value (the less the better): '+str(upgrade_value))



    queue.close()
    print('best cost: '+str(min_cost))
    print('path: '+str(best_path))


    #формируем путь с именами ячеек
    real_path = [int(columns[v]) for v in best_path]


    result={'path': real_path, 'cost':min_cost}
    with io.open(file_result, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False))