import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# import tensorflow.keras as kr
import time

# from NN_FUNCTIONS import check, separate_date, num_parametros
# from NN_FUNCTIONS import optimize_eprochs_neural_network, optimize_complete_neural_network
from NN_FUNCTIONS import optimize_complete_neural_network


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
    

sc = StandardScaler()

y = states['GROUP_NUMBER']
X = sc.fit_transform(states_norm)

# HYPERPARAMETERS

SEED_SPLIT = 0
EPROCHS = 1200

# BEST_ARQUITECTURE, BEST_ACC_TRAIN, BEST_ACC_VAL, BEST_EPROCH  = optimize_complete_neural_network(states_norm, y, val_size = 0.3,
                                                                                                  # eprochs = EPROCHS, layers = [6],
                                                                                                  # seed_split = SEED_SPLIT,
                                                                                                  # name_txt = 'Results/best_arquitecture_states_val_layers_6_850_fin.txt')

# BEST_ARQUITECTURE, BEST_ACC_TRAIN, BEST_ACC_VAL, BEST_EPROCH  = optimize_complete_neural_network(states_norm, y, val_size = 0.3,
#                                                                                                  eprochs = EPROCHS, layers = [3,4,5],
#                                                                                                  seed_split = SEED_SPLIT,
#                                                                                                  name_txt = 'Results/best_arquitecture_states_val_layers_3,4,5_binary.txt',
#                                                                                                  fun_act_output = 'sigmoid')




fin = time.time()
print(fin-inicio)