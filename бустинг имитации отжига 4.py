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
#import io
#import json
from pytsp.christofides_tsp import christofides_tsp
import time
#----------------------------------------------

import time
start_time = time.time()
from math import sin, cos, sqrt, atan2, radians
from scipy.spatial.distance import pdist, squareform

class ReadData():
    def __init__(self, filename):

        self.name = filename[:-4]
        self.size = self.getSize()
        self.EdgeWeightType = self.getEdgeWeightType()
        self.format_ = self.getFormat()  # for EXPLICIT data only
        self.time_to_read = 0

    def getFormat(self):
        format_ = "None"
        try:
            with open(f'{self.name}.tsp') as data:#TSP_Data/
                datalist = data.read().split()
                for ind, elem in enumerate(datalist):
                    if elem == "EDGE_WEIGHT_FORMAT:":
                        format_ = datalist[ind + 1]
                        break
                    elif elem == "EDGE_WEIGHT_FORMAT":
                        format_ = datalist[ind + 2]
                        break
            return format_

        except IOError:
            print("Input file "+self.name+" not found")
            sys.exit(1)

    def getEdgeWeightType(self):
        EdgeType = "None"
        try:
            with open(f'{self.name}.tsp') as data:#TSP_Data/
                datalist = data.read().split()
                for ind, elem in enumerate(datalist):
                    if elem == "EDGE_WEIGHT_TYPE:":
                        EdgeType = datalist[ind + 1]
                        #print(EdgeType)
                        break
                    elif elem == "EDGE_WEIGHT_TYPE":
                        EdgeType = datalist[ind + 2]
                        #print(EdgeType)
                        break
            return EdgeType

        except IOError:
            print("Input file "+self.name+" not found")
            sys.exit(1)

    def getSize(self):
        """
        Return size of instances (i.e. Number of
        cities)

        """
        size = 0
        try:
            with open(f'{self.name}.tsp') as data:#TSP_Data/
                datalist = data.read().split()
                for ind, elem in enumerate(datalist):
                    if elem == "DIMENSION:":
                        size = datalist[ind + 1]
                        #print(size)
                        break
                    elif elem == "DIMENSION":
                        size = datalist[ind + 2]
                        #print(size)
                        break
            return int(size)
        except IOError:
            print("Input file "+self.name+" not found")
            sys.exit(1)

    def read_Data(self):
        with open(f'{self.name}.tsp') as data:#TSP_Data/
            cities = []
            Isdata = True
            while (Isdata):
                line = data.readline().split()
                if len(line) <= 0:
                    break
                tempcity = []
                for i, elem in enumerate(line):
                    try:
                        temp = float(elem)
                        tempcity.append(temp)
                    except ValueError:
                        break
                if len(tempcity) > 0:
                    cities.append(np.array(tempcity))
        return np.array(cities)

    def GetDistanceMat(self):
        if self.EdgeWeightType == "EXPLICIT":
            DistanceMat = self.getMat()
            self.time_to_read = time.time() - start_time
            return DistanceMat
        elif self.EdgeWeightType == "EUC_2D" or "CEIL_2D":
            DistanceMat = self.EuclidDist()
            self.time_to_read = time.time() - start_time
            return DistanceMat
        elif self.EdgeWeightType == "GEO":
            DistanceMat = self.GeographicDist()
            self.time_to_read = time.time() - start_time
            return DistanceMat
        else:
            return None

    def EuclidDist(self):
        cities = self.read_Data()
        #DistanceDict = {}
        A = cities[:, 1:3]
        DistanceMat = np.round(squareform(pdist(A)))
        return DistanceMat

    def GeographicDist(self):
        a = time.time()
        R = 6373.0
        cities = self.read_Data()
        DistanceMat = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(0, i + 1):
                node1 = cities[i]
                node2 = cities[j]
                lat1 = radians(node1[1])
                lat1 = radians(node1[1])
                lon1 = radians(node1[2])
                lat2 = radians(node2[1])
                lon2 = radians(node2[2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                distance = R * c
                DistanceMat[i, j] = distance
                DistanceMat[j, i] = distance

        return DistanceMat

    def getMat(self):
        DataFormat = self.getFormat()
        if DataFormat == "FULL_MATRIX":
            cities = self.read_Data()
            DistanceMat = cities[:self.size]
            return DistanceMat

        elif DataFormat == "LOWER_DIAG_ROW":
            with open(f'{self.name}.tsp') as file:#TSP_Data/
                indicator = False
                data = file.read().split()
                templist = []
                cities = []
                for elem in data:
                    if elem == "EDGE_WEIGHT_SECTION":
                        indicator = True
                        continue
                    if indicator:
                        try:
                            it = float(elem)
                            templist.append(it)
                        except:
                            break
                        if it == 0:
                            cities.append(templist)
                            templist = []
                DistanceMat = np.zeros((self.size, self.size))
                for i in range(self.size):
                    temp = []
                    l = len(cities[i])
                    for j in range(self.size):
                        if j <= (l - 1):
                            temp.append(cities[i][j])
                        else:
                            temp.append(cities[j][i])
                    DistanceMat[i] = temp
                return DistanceMat
        elif DataFormat == "UPPER_DIAG_ROW":
            with open(f'{self.name}.tsp') as file:#TSP_Data/
                indicator = False
                data = file.read().split()
                templist = []
                cities = []
                for elem in data:
                    if elem == "EDGE_WEIGHT_SECTION":
                        indicator = True
                        continue
                    if indicator:
                        try:
                            it = float(elem)
                            templist.append(it)
                        except ValueError:
                            break
                        if it == 0:
                            cities.append(templist)
                            templist = []
                print("Need to complete it")
        else:
            sys.exit("No Format Match for EXPLICIT data")





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
    if not match:
        raise ValueError("Нулевой iter_limit")
    iter_limit=int(match[1])




    r = ReadData(file)
    graph=r.GetDistanceMat()
#
# =============================================================================



    #начальный путь
    initial_path = christofides_tsp(graph)
    initial_min_cost=route_cost(graph, initial_path)
    total_lim=len(initial_path)
    print('initial cost: '+str(initial_min_cost))

    queue = Queue()

    size=len(initial_path)#размер пути
    alpha=1-1e-3#параметр уменьшения температуры


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
