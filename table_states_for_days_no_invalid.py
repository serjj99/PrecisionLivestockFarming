import pandas as pd
import numpy as np

"""
Tabla de los states de cada animal de cada grupo cada dia (poner una tolerancia minutos).
"""


df = pd.read_csv('Dates/Tag_Data.csv')


df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([12], 11)


frec = df['STATE_NUMBER'].value_counts()

animals = df['ANIMAL_NUMBER'].unique()
times = df['START_TIME_LOCAL']

days = []
for time in times:
    days.append(time[:10])  # quitamos las horas

df['DAYS'] = days


# ARRAY STATES:
# Matriz (len(animales que pertenecen a un grupo) x 15)
# 1ra col: numero animal
# 2da col: grupo animal
# 3ra-13a cols: duracion que ha estado en un estado
# 14a col: suma de todas las veces que ha estado en cualquier estado

tol_time = int(60*12) # 12 horas (en minutos)

# array_states = np.zeros((3190,15))
array_states = np.zeros((2021,15))


n_a = 0
for a in animals:
    for g in [1,3]:
        days_animal = df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g)]['DAYS']
        
        if days_animal.count() != 0:
            days_for_animal = days_animal.unique()
            
            for d in days_for_animal:
                n_group = df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g) & (df['DAYS']==d)]['DURATION'].sum() - df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g) & (df['DAYS']==d) & (df['STATE_NUMBER']==15)]['DURATION'].sum()
                
                if n_group >= tol_time:
                    array_states[n_a,0] = int(a)
                    array_states[n_a,1] = int(g)
                    
                    for n in range(2,13):
                        #Se añade el numero de veces que hace el animal 'a' cuando se encuentra en el grupo 'g' el estado 'n-1'
                        array_states[n_a,n] = df[(df['ANIMAL_NUMBER']==a) & (df['GROUP_NUMBER']==g) & (df['DAYS']==d) & (df['STATE_NUMBER']==n-1)]['DURATION'].sum()

                    #En la última columna se añade la suma de todos los estados que ha hecho cada animal cada dia, es decir, el tamaño de cada 'animal'
                    array_states[n_a,-1] = sum(array_states[n_a,2:])
                    n_a+=1



#ANIMAL_NUMBER  GROUP_NUMBER  STATE_NUMBER_1 ... STATE_NUMBER_12  STATE_SUMS  STATE_NUMBER_1_NORM ... STATE_NUMBER_12_NORM

states = pd.DataFrame()

states['ANIMAL_NUMBER'] = array_states[:,0]

states['GROUP_NUMBER'] = array_states[:,1]

# for i in range(2,14):
#     states['STATE_NUMBER_{}'.format(i-1)] = array_states[:,i]
    
# for i in range(2,14):
#     states['STATE_DURATION_{}'.format(i-1)] = array_states[:,i+12]

for i in range(2,13):
    states['STATE_DURATION_{}'.format(i-1)] = array_states[:,i]

states['STATE_SUMS'] = array_states[:,-1]

# for i in range(2,14):
#     states['STATE_NUMBER_NORM_{}'.format(i-1)] = array_states[:,i+12] / states['STATE_SUMS']

for i in range(2,13):
    states['STATE_NUMBER_NORM_{}'.format(i-1)] = array_states[:,i] / states['STATE_SUMS']



#SAVE

states.to_csv('Dates/states_for_days_no_invalid_1_3.csv')
states.to_excel('Dates/excel/states_for_days_no_invalid_1_3.xlsx')