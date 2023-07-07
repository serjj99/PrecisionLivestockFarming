import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score, mean_squared_error
import tensorflow as tf
import tensorflow.keras as kr
import keras_tuner as kt

from NN_FUNCTIONS import check, compare_values, separate_date, num_parametros, optimize_eprochs_neural_network, cross_validation_neural_network



seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# DATES

states = pd.read_csv('Dates/states_for_days_no_invalid.csv')
# states = pd.read_csv('Dates/states_no_invalid.csv')

states_norm = pd.DataFrame()
states_norm['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
# for i in range(3,14):
#     states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]  

sc = StandardScaler()

y = states['GROUP_NUMBER']
X = sc.fit_transform(states_norm)
# X = np.array([states_norm])[0]



# CROSS VALIDATION

p = X.shape[1]

eprochs = 526  # num of epochs to test for
neurons = [p, 18, 9, 6, 6, 3]  # Number of neurons
lr = 1e-2   # learning rate

cv = 5

# acc_train_cv = []
# acc_test_cv = []

# for i in range(cv):
#     X_train, X_test, y_train, y_test = separate_date(states_norm, y, i, cv)
    
#     sc = StandardScaler() 
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
    
#     y_train = kr.utils.to_categorical(y_train-1)
#     y_test = kr.utils.to_categorical(y_test-1)
    
#     best_eproch, best_acc_train, best_acc_test, acc_train, acc_test = optimize_eprochs_neural_network(X_train, y_train, X_test, y_test,
#                                                                             neuron=nn, eprochs=NB_EPOCHS, write = False) 
    
#     acc_train_cv.append(acc_train[-1])
#     acc_test_cv.append(acc_test[-1])

acc_train_cv_all, acc_test_cv_all, acc_train_cv, acc_test_cv, acc_train_cv_std, acc_test_cv_std, models_cv = cross_validation_neural_network(states_norm, y, neurons, eprochs, 
                                                                                                                                              cv = cv, LR = lr)

print('Accuracy train cv =', acc_train_cv *100, '%')
print('Std accuracy train cv =', acc_train_cv_std, '\n')

print('Accuracy test cv =', acc_test_cv *100, '%')
print('Std accuracy test cv =', acc_test_cv_std)
