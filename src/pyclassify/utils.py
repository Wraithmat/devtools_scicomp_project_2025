import os
import yaml
from numba.pycc import CC

@profile
def distance(point1: list[float], point2: list[float]):
    distance=0
    for i in range(len(point1)):
        distance+=(point1[i]-point2[i])**2
    return distance**(1/2)

@profile
def distance_numpy(point1, point2):
    return ((point1-point2)**2).sum()**(0.5)


cc = CC('module')
@profile
@cc.export('distance_numba','f8(f8[:], f8[:])')
def distance_numba(point1,point2):
    distance=0
    for i in range(len(point1)):
        distance+=(point1[i]-point2[i])**2
    return distance**(1/2)

cc.compile()

@profile
def majority_vote(neighbors: list[int]):
    classes=list(set(neighbors))
    counted=[]
    max_val=0
    best_index=0
    i=-1
    for el in classes:
        i+=1
        counted.append(neighbors.count(el))
        if counted[-1]>max_val:
            max_val=counted[-1]
            best_index=i
    return classes[best_index]

def read_config(file):
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath,'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs

def read_file(file_path):
    with open(file_path) as f:
        lis = [line.split(',') for line in f]     
        features = [[float(value) for value in el[:-1]] for el in lis]
        labels = [el[-1] for el in lis]
    return features, labels
                
