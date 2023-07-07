# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

import tensorflow as tf
# import tensorflow.keras as kr
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.models import Sequential

from RNN_function import RNN_LSTM_animals


pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

name_txt='Results/RNN_prediction.txt'
f = open(name_txt, "w")
f.write('categorical_accuracy \t val_categorical_accuracy \t Accuracy \t CM \n')
f.close()


# GRUPO 1/3
# df1 = pd.read_csv('Dates/Dates_times/1/states_times_animal_1379.csv', index_col='Unnamed: 0')
# df2 = pd.read_csv('Dates/Dates_times/1/states_times_animal_1524.csv', index_col='Unnamed: 0')
# df3 = pd.read_csv('Dates/Dates_times/1/states_times_animal_1542.csv', index_col='Unnamed: 0')
# df4 = pd.read_csv('Dates/Dates_times/1/states_times_animal_1778.csv', index_col='Unnamed: 0')
# df5 = pd.read_csv('Dates/Dates_times/1/states_times_animal_1823.csv', index_col='Unnamed: 0')

# RNN_LSTM_animals(df1, min_pred=1440, grupo='1', animal='1379', name_txt=name_txt)
# RNN_LSTM_animals(df2, min_pred=1440, grupo='1', animal='1524', name_txt=name_txt)
# RNN_LSTM_animals(df3, min_pred=1440, grupo='1', animal='1542', name_txt=name_txt)
# RNN_LSTM_animals(df4, min_pred=1440, grupo='1', animal='1778', name_txt=name_txt)
# RNN_LSTM_animals(df5, min_pred=1440, grupo='1', animal='1823', name_txt=name_txt)


# GRUPO 2
# df6 = pd.read_csv('Dates/Dates_times/2/states_times_animal_1379.csv', index_col='Unnamed: 0')
# df7 = pd.read_csv('Dates/Dates_times/2/states_times_animal_1524.csv', index_col='Unnamed: 0')
df8 = pd.read_csv('Dates/Dates_times/2/states_times_animal_1542.csv', index_col='Unnamed: 0')
# df9 = pd.read_csv('Dates/Dates_times/2/states_times_animal_1778.csv', index_col='Unnamed: 0')
df10 = pd.read_csv('Dates/Dates_times/2/states_times_animal_1823.csv', index_col='Unnamed: 0')

# RNN_LSTM_animals(df6, min_pred=1440, grupo='2', animal='1379', name_txt=name_txt)
# RNN_LSTM_animals(df7, min_pred=1440, grupo='2', animal='1524', name_txt=name_txt)
RNN_LSTM_animals(df8, min_pred=1440, grupo='2', animal='1542', name_txt=name_txt)
# RNN_LSTM_animals(df9, min_pred=1440, grupo='2', animal='1778', name_txt=name_txt)
RNN_LSTM_animals(df10, min_pred=1440, grupo='2', animal='1823', name_txt=name_txt)

