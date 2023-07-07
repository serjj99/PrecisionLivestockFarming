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

from NN_FUNCTIONS import check, compare_values, separate_date, num_parametros, optimize_eprochs_neural_network, cross_validation_neural_network, predictions_models_cv



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
y = y.replace([3], 1)
# y -= 1
X = sc.fit_transform(states_norm)

# p = X.shape[1]

# X_train, X_val, y_train, y_val = train_test_split(states_norm, y-1, test_size = 0.25, random_state = 0)


# X_train = sc.fit_transform(X_train)
# X_val = sc.transform(X_val)


# y_train_ori = y_train
# y_val_ori = y_val
# y_train = kr.utils.to_categorical(y_train-1)
# y_val = kr.utils.to_categorical(y_val-1)


#Variables
# p = X.shape[1]
# max_param = X_train.shape[0]/3

# NB_EPOCHS = 50
# nn = [p, 16, 8, 2, 1]  # Number of neurons
# lr = 2e-2   # learning rate



# np.random.seed(42)
# tf.random.set_seed(42)
    
# model = kr.Sequential()

# for n_neu in nn[1:-1]:
#     model.add(kr.layers.Dense(n_neu, activation='relu'))
# model.add(kr.layers.Dense(nn[-1], activation='sigmoid'))
    
# model.compile(loss='mse', optimizer=kr.optimizers.SGD(learning_rate = lr), metrics = 'acc')
   
# history = model.fit(X_train, y_train, epochs=NB_EPOCHS, validation_data=(X_val, y_val))

# print(model.summary())



# plt.figure(figsize=(8,8))
# plt.title('ACCURACY')
# plt.plot(history.history['acc'], color='blue', label='Train acc')
# plt.plot(history.history['val_acc'], color='red', label='Test acc')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# # plt.savefig('Images/acc.png')
# plt.show()




# # PLOTS

# # plt.figure(figsize=(8,8))
# # # plt.title('ACCURACY')
# # # plt.plot(acc_, color='blue', label='Train acc')
# # # plt.plot(val_acc_, color='red', label='Test acc')
# # plt.plot(history.history['acc'], color='blue', label='Train acc')
# # plt.plot(history.history['val_acc'], color='red', label='Test acc')
# # plt.legend()
# # plt.xlabel('Epochs')
# # plt.ylabel('Acc')
# # # plt.savefig('Images/acc.png')
# # plt.show()

# plt.figure(figsize=(8,8))
# # plt.title('ACCURACY')
# # plt.plot(acc_, color='blue', label='Train acc')
# # plt.plot(val_acc_, color='red', label='Test acc')
# plt.plot(history.history['categorical_accuracy'], color='blue', label='Train categorical_accuracy')
# plt.plot(history.history['val_categorical_accuracy'], color='red', label='Test categorical_accuracy')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Categorical_accuracy')
# # plt.savefig('Images/acc.png')
# plt.show()

# plt.figure(figsize=(8,8))
# # plt.title('LOSS (mean squared error)')
# # plt.plot(loss_, color='blue', label='Train loss')
# # plt.plot(val_loss_, color='red', label='Test loss')
# plt.plot(history.history['loss'], color='blue', label='Train mse cv')
# plt.plot(history.history['val_loss'], color='red', label='Test mse cv')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss (mean squared error)')
# # plt.savefig('Images/loss_cv.png')
# plt.show()



# RESULTS

p = X.shape[1]

eprochs = 88
neurons = [p, 6, 12, 1]  # Number of neurons
# neurons = [p, 24, 15, 6, 1]  # Number of neurons
lr = 2e-2   # learning rate

num_parametros(neurons)

acc_train_cv_all, acc_test_cv_all, acc_train_cv, acc_test_cv, acc_train_cv_std, acc_test_cv_std, models_cv = cross_validation_neural_network(states_norm, y, neurons, eprochs, 
                                                                                                                                              cv = 5,
                                                                                                                                  fun_act_hidden = 'relu', 
                                                                                                                                  fun_act_output = 'sigmoid', 
                                                                                                                                  LOSS = 'mse', METRICS = 'acc', LR = lr)

# acc_train_cv_one, acc_test_cv_one = cross_validation_neural_network(states_norm, y, neurons, eprochs, all_history = True, plot = True, cv = 5, fun_act_hidden = 'relu', 
#                                                             fun_act_output = 'sigmoid', LOSS = 'mse', METRICS = 'acc', LR = lr)


# np.where(acc_test_cv_one == max(acc_test_cv_one))


y_pred = predictions_models_cv(X, models_cv)
cm = confusion_matrix(y-1, y_pred)

modelo = models_cv[0]
for i in range(len(modelo.get_weights()[0])):
    suma = 0
    for j in range(len(modelo.get_weights()[0][i])):
        suma += abs(modelo.get_weights()[0][i][j])
    print('Neurona {} = {}'.format(i+1,suma))


print('Accuracy train cv =', acc_train_cv *100, '%')
print('Std accuracy train cv =', acc_train_cv_std, '\n')

print('Accuracy test cv =', acc_test_cv *100, '%')
print('Std accuracy test cv =', acc_test_cv_std, '\n')


print('CM =', cm)
print('Acc total cv =', (sum(np.diag(cm)) / (sum(sum(cm)))) *100, '%')
