# Igual que table1 pero sin el 15 (invalid).
# En este caso el state 12 equivale al state 11.

import pandas as pd
import numpy as np


df = pd.read_csv('Dates/Tag_Data.csv')

df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([12], 11)


frec = df['STATE_NUMBER'].value_counts()

animals = df['ANIMAL_NUMBER'].unique()



# ARRAY STATES:
# Matriz (len(animales que pertenecen a un grupo) x 15)
# 1ra col: numero animal
# 2da col: grupo animal
# 3ra-13a cols: duracion que ha estado en un estado
# 14a col: suma de todas las veces que ha estado en cualquier estado

array_states = np.zeros((113,14)) # 113 animales que pertenecen a un grupo

n_a = 0
for a in animals:
    for g in range(1,4):
        n_group = df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g)]['GROUP_NUMBER'].count() - df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g) & (df['STATE_NUMBER']==15)]['DURATION'].sum()
        
        if n_group != 0:
            array_states[n_a,0] = int(a)
            array_states[n_a,1] = int(g)
            
            for n in range(2,13):
                #Se añade el numero de veces que hace el animal 'a' cuando se encuentra en el grupo 'g' el estado 'n-1'
                array_states[n_a,n] = df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g) & (df['STATE_NUMBER']==n-1)]['DURATION'].sum()
      
            #En la última columna se añade la suma de todos los estados que ha hecho cada animal, es decir, el tamaño de cada 'animal'
            array_states[n_a,-1] = sum(array_states[n_a,2:])
            n_a+=1





#ANIMAL_NUMBER  GROUP_NUMBER  STATE_NUMBER_1 ... STATE_NUMBER_12  STATE_SUMS  STATE_NUMBER_1_NORM ... STATE_NUMBER_12_NORM

states = pd.DataFrame()

states['ANIMAL_NUMBER'] = array_states[:,0]

states['GROUP_NUMBER'] = array_states[:,1]

for i in range(2,13):
    states['STATE_DURATION_{}'.format(i-1)] = array_states[:,i]


states['STATE_SUMS'] = array_states[:,-1]

for i in range(2,13):
    states['STATE_NUMBER_NORM_{}'.format(i-1)] = array_states[:,i] / states['STATE_SUMS']


# states_norm = pd.DataFrame()
# states_norm['S_1'] = states['STATE_NUMBER_NORM_1']
# for i in range(3,14):
#     states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]

#SAVE

states.to_csv('Dates/states_no_invalid.csv')
states.to_excel('Dates/excel/states_no_invalid.xlsx')
# states_norm.to_excel('Dates/states_norm.xlsx')






