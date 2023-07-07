import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as kr


sns.set(style='whitegrid', palette='muted', font_scale=1.5)
plt.rcParams['figure.figsize'] = 22,10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


days_week = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

df = pd.read_csv('Dates/states_times_2.csv', parse_dates=['START_TIME_LOCAL'], index_col='START_TIME_LOCAL')
# df = pd.read_csv('Dates/states_times_1_3_agrup.csv', parse_dates=['START_TIME_LOCAL'], index_col='START_TIME_LOCAL')

# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([4], 2)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([5], 2)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([6], 3)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([7], 3)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([8], 3)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([9], 1)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([10], 2)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([11], 2)



def plot_states_for_day_ALL(df, div = 10, savename = 'Results/Plots_times_seq/2/Gr2'):
    animals = df['ANIMAL_NUMBER'].unique()
    
    for i,a in enumerate(animals):
        # if i%div == 0:
        if i == 20:
            df_for_animal = df[df['ANIMAL_NUMBER'] == a]
            days_for_animal = df_for_animal['DAY'].unique()
            len_days = len(days_for_animal)
            if len_days >= 7:
                days_for_animal = days_for_animal[int(len_days/2 - 3.5) : int(len_days/2 + 3.5)]
            for d in days_for_animal:
                df_for_animal_days = df_for_animal[df_for_animal['DAY'] == d]
                day_of_week = df_for_animal_days['DAY_OF_WEEK'][0]
                plt.figure()
                sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df_for_animal_days) #.set(title='Animal {} & day {}'.format(a,d))
                # if (day_of_week == 5) or (day_of_week == 6):
                #     sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df_for_animal_days, color='red') #.set(title='Animal {} & day {}'.format(a,d))
                # else:
                #     sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df_for_animal_days) #.set(title='Animal {} & day {}'.format(a,d))
                fig = sns_plot.get_figure()
                fig.savefig(savename + '_Animal_' + str(a) + '_Day_' + str(d) + '_' + days_week[day_of_week] + '.png')
                    



def plot_states_for_day_SELECTION(df, days, years, savename = 'Results/Plots_times_seq/2/Gr2'):
    animals = df['ANIMAL_NUMBER'].unique()

    for i,a in enumerate(animals):
        df_for_animal = df[df['ANIMAL_NUMBER'] == a]
        days_for_animal = df_for_animal['DAY'].unique()
        for d in days_for_animal:
            if d in days:
                df_for_animal_days = df_for_animal[df_for_animal['DAY'] == d]
                year_of_day = df_for_animal_days['YEAR'].unique()[0]
                if year_of_day in years:
                    day_of_week = df_for_animal_days['DAY_OF_WEEK'][0]
                    plt.figure()
                    sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df_for_animal_days) #.set(title='Animal {} & day {}'.format(a,d))
                    fig = sns_plot.get_figure()
                    fig.savefig(savename + '_Animal_' + str(a) + '_Day_' + str(d) + '_' + days_week[day_of_week] + '.png')
                        
                        


                    

                    
# years = [2021]    
# days = [27]           

# plot_states_for_day_ALL(df)

# plot_states_for_day_SELECTION(df, days, years) #, savename = 'Results/Plots_times_seq/1_3/Gr1_3'


df_complete = pd.read_csv('Dates/states_times.csv', index_col='Unnamed: 0')


