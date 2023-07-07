import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# import tensorflow.keras as kr
import time

# from NN_FUNCTIONS import check, separate_date, num_parametros
# from NN_FUNCTIONS import optimize_eprochs_neural_network, optimize_complete_neural_network
from NN_FUNCTIONS import optimize_complete_neural_network, optimize_complete_cv_neural_network


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
inicio = time.time()


# DATES

states = pd.read_csv('Dates/states_for_days_no_invalid.csv')

states_norm = pd.DataFrame()
states_norm['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
# for i in range(3,14):
#     states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]  

sc = StandardScaler()

y = states['GROUP_NUMBER']
y = y.replace([3], 1)
# y -= 1
X = sc.fit_transform(states_norm)


#Variables
NB_EPOCHS = 300
lr = 2e-2   # learning rate


#Optimization

# BEST_ARQUITECTURE, BEST_ACC_TRAIN, BEST_ACC_VAL, BEST_EPROCH = optimize_complete_neural_network(states_norm, y, val_size = 0.25, eprochs = NB_EPOCHS, layers = [3,4,5],
#                                                                                                 name_txt = 'Results/best_arquitecture_states_binary_layers_3_4_5.txt', 
#                                                                                                 output_layer = 1, fun_act_hidden = 'relu', fun_act_output = 'sigmoid',
#                                                                                                 LOSS = 'mse', METRICS = 'acc', LR = lr)


BEST_ARQUITECTURE, BEST_ACC_TRAIN, BEST_ACC_VAL, BEST_EPROCH = optimize_complete_cv_neural_network(states_norm, y, cv = 5, eprochs = NB_EPOCHS, layers = [5,6],
                                                                                                   name_txt = 'Results/BEST_ARQ_BINARY_CV_5/best_arquitecture_CV5_states_binary.txt',
                                                                                                   output_layer = 1, fun_act_hidden = 'relu', fun_act_output = 'sigmoid',
                                                                                                   LOSS = 'mse', METRICS = 'acc', LR = lr)



fin = time.time()
print(fin-inicio)