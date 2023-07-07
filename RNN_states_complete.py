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


# df = pd.read_csv('Dates/Dates_times/1/states_times_animal_1378.csv', index_col='Unnamed: 0')
# df = pd.read_csv('Dates/states_times_1_3_agrup.csv', parse_dates=['START_TIME_LOCAL'], index_col='START_TIME_LOCAL')

# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([4], 2)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([5], 2)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([6], 3)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([7], 3)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([8], 3)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([9], 1)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([10], 2)
# df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([11], 2)



def plot_states_for_day_ALL(df, div = 10, savename = 'Results/Plots_times_seq/2/Gr2'): #savename = 'Results/Plots_times_seq/1_3/Gr1_3'
    animals = df['ANIMAL_NUMBER'].unique()
    
    for i,a in enumerate(animals):
        # if i%div == 0:
        if i == 0:
            df_for_animal = df[df['ANIMAL_NUMBER'] == a]
            days_for_animal = df_for_animal['DAY'].unique()
            # len_days = len(days_for_animal)
            # if len_days >= 7:
            #     days_for_animal = days_for_animal[int(len_days/2 - 3.5) : int(len_days/2 + 3.5)]
            for d in days_for_animal:
                df_for_animal_days = df_for_animal[df_for_animal['DAY'] == d]
                plt.figure()
                sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df_for_animal_days) #.set(title='Animal {} & day {}'.format(a,d))
                plt.xticks(np.linspace(0,1440,25))
                fig = sns_plot.get_figure()
                fig.savefig(savename + '_Animal_' + str(a) + '_Day_' + str(d) + '.png')
                    



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
                    plt.figure()
                    sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df_for_animal_days) #.set(title='Animal {} & day {}'.format(a,d))
                    plt.xticks(np.linspace(0,1440,25))
                    fig = sns_plot.get_figure()
                    fig.savefig(savename + '_Animal_' + str(a) + '_Day_' + str(d) + '.png')
                        
                        


def plot_states_ALL(df, animal, savename = 'Results/Plots_times_seq/1_3/Gr1_3'): #savename = 'Results/Plots_times_seq/1_3/Gr1_3'
    days = df['DAY'].unique()
    steps = int((4*(len(days)-1))+1)
    max_minutes = len(df) + (1440-len(df)%1440)
    plt.figure(figsize=(int(8*len(days)), 8))
    sns_plot = sns.pointplot(x='MINUTES', y='STATE_NUMBER', data=df) #.set(title='Animal {} & day {}'.format(a,d))
    plt.xticks(np.linspace(0, max_minutes, steps))
    fig = sns_plot.get_figure()
    fig.savefig(savename + '_Animal' + str(animal) + '.png')                

                    
         


df = pd.read_csv('Dates/Dates_times/2/states_times_animal_1379.csv', index_col='Unnamed: 0')
plot_states_ALL(df, 1780, savename = 'Results/Plots_times_seq/2/Gr2')
