import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data_ori = pd.read_csv('Dates/Tag_Data.csv')

States_times_1 = pd.read_csv('Dates/states_times_1.csv')
States_times_2 = pd.read_csv('Dates/states_times_2.csv')
States_times_3 = pd.read_csv('Dates/states_times_3.csv')

States_for_days = pd.read_csv('Dates/states_for_days_no_invalid.csv')



animals = Data_ori['ANIMAL_NUMBER'].unique()

animals1 = States_times_1['ANIMAL_NUMBER'].unique()
animals2 = States_times_2['ANIMAL_NUMBER'].unique()
animals3 = States_times_3['ANIMAL_NUMBER'].unique()



animals_days = States_for_days['ANIMAL_NUMBER'].unique()

frec_group = States_for_days['GROUP_NUMBER'].value_counts()

cattle_id = Data_ori['CATTLE_ID'].unique()




states_norm = pd.DataFrame()
states_norm['S_1'] = States_for_days['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm['S_{}'.format(i-1)] = States_for_days['STATE_NUMBER_NORM_{}'.format(i-1)]
    
states_norm[States_for_days['Unnamed: 0'] == 2245]


plt.figure(figsize=(10, 6))
plt.bar([1,2,3,4,5,6,7,8,9,10,11], [0.35833333333333334,0.12430555555555556,0.044444444444444446,0.2076388888888889,0.15,0.004166666666666667,0.04375,0.06736111111111111,0.0,0.0,0.0])
plt.xticks(np.linspace(0, 11, 12))
plt.ylim((0,1))
plt.show()


plt.figure(figsize=(10, 6))
plt.bar([1,2,3,4,5,6,7,8,9,10,11], [0.6785714285714286,0.1488095238095238,0.0011904761904761906,0.0,0.0,0.0,0.0,0.09642857142857143,0.0,0.02142857142857143,0.05357142857142857])
plt.xticks(np.linspace(0, 11, 12))
plt.ylim((0,1))
plt.show()


plt.figure(figsize=(10, 6))
plt.bar([1,2,3,4,5,6,7,8,9,10,11], [0.5486111111111112,0.15347222222222223,0.004861111111111111,0.18680555555555556,0.008333333333333333,0.0,0.0,0.010416666666666666,0.0,0.009722222222222222,0.07777777777777778])
plt.xticks(np.linspace(0, 11, 12))
plt.ylim((0,1))
plt.show()



s1 = States_for_days['STATE_NUMBER_NORM_1']
s2 = States_for_days['STATE_NUMBER_NORM_2']
s3 = States_for_days['STATE_NUMBER_NORM_3']
grupo = States_for_days['GROUP_NUMBER']

colors = ['r', 'g', 'b']

c_list = []
for g in grupo:
    c_list.append(colors[int(g-1)])
    
    
ax = plt.figure(figsize=(12,12)).add_subplot(projection='3d')

ax.scatter(s1, s2, s3, c=c_list)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
ax.set_xlabel('Baja actividad')
ax.set_ylabel('Media actividad')
ax.set_zlabel('Alta actividad')

plt.show()