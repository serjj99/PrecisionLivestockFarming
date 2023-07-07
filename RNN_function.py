import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


pd.options.mode.chained_assignment = None
tf.random.set_seed(0)



def RNN_LSTM_animals(df, min_pred=1440, grupo='0', animal='0', name_txt='Results/RNN_prediction.txt'):
    
    print('START: grupo: ' + grupo + ', animal: ' + animal)
    
    # f = open(name_txt+'_animal_'+animal+'_grupo_'+grupo+'.txt', "w")
    # f.write('categorical_accuracy \t val_categorical_accuracy \t Accuracy \t CM \n')
    # f.close()
    
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([4], 2)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([5], 2)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([6], 3)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([7], 3)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([8], 3)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([9], 1)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([10], 2)
    df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([11], 2)
    
    
    y_ori = df['STATE_NUMBER']
    y = y_ori.values.reshape(-1, 1)
    # y = kr.utils.to_categorical(y-min(y))
    
    
    # y_start_predict = int(len(y) - 1*60)  # = n_forecast
    y_start_predict = int(len(df) - min_pred)
    # y_min = int(y_start_predict - 3*1440)
    y_min = 0
    
    
    n_lookback = int(min_pred)  # 6*60  length of input sequences (lookback period)  1 day
    n_forecast = int(min_pred)  # length of output sequences (forecast period)  1 day
    
    
    y_actual = y[y_min:y_start_predict]
    y_to_predict = y[y_start_predict:y_start_predict+n_forecast]
    
    
    y_actual_CAT = kr.utils.to_categorical(y_actual-min(y))
    y_to_predict_CAT = kr.utils.to_categorical(y_to_predict-min(y))
    
    X = []
    Y = []
    
    for i in range(n_lookback, len(y_actual_CAT) - n_forecast + 1):
        X.append(y_actual_CAT[i - n_lookback: i])
        Y.append(y_actual_CAT[i: i + n_forecast])
    
    X = np.array(X)
    Y = np.array(Y)
    
    
    X_to_predict = y_actual_CAT[- n_lookback:]  # last available input sequence
    X_to_predict = X_to_predict.reshape(1, n_lookback, df.shape[1])
    
    
    # fit the model
    # #El significado de las 3 dimensiones de entrada son: muestras, pasos de tiempo y características
    
    model = Sequential()
    # model.add(Embedding(360, 128))
    # model.add(Dropout(0.2))
    model.add(LSTM(units=128, dropout=0.2, return_sequences=True, input_shape=(n_lookback, df.shape[1])))  #input_shape toma una tupla de dos valores que definen la cantidad de pasos de tiempo y características
    model.add(Dense(64, activation='relu'))
    model.add(Dense(df.shape[1], activation='softmax')) #n_forecast
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = 'categorical_accuracy')
    history = model.fit(X, Y, epochs=3, batch_size=32, verbose=0, validation_data=(X_to_predict, y_to_predict_CAT.reshape(1, n_forecast, df.shape[1])))
    
    
    # print(history.history['categorical_accuracy'])
    # print(history.history['val_categorical_accuracy'])
    
    # generate the forecasts
    
    
    
    Y_predict = model.predict(y_to_predict_CAT.reshape(1, n_forecast, df.shape[1]))
    Y_predict = np.argmax(Y_predict[0], axis=1) + 1
    
    
    # print('y_real = {}'.format(y_to_predict))
    # print('y_predict = {}'.format(Y_predict))
    
    
    y_real = np.array([])
    for date_real in y_to_predict:
        y_real = np.append(y_real, date_real[0])
    
    good=0
    bad=0
    for i in range(len(y_real)):
        if y_real[i] == Y_predict[i]:
            good+=1
        else:
            bad+=1
    
    accuracy = good / (good+bad)
    # print('Accuracy = {}%'.format(accuracy*100))
    
    cm = confusion_matrix(y_real, Y_predict)
    # print('CM = {}'.format(cm))
    
    f = open(name_txt, "a")
        
    f.write(str(history.history['categorical_accuracy'][-1]) + '\t' + str(history.history['val_categorical_accuracy'][-1]) + '\t' + str(round(accuracy*100, 6)) + '\t' + '[[' + str(cm[0,0]) + ',' + str(cm[0,1]) + ',' + str(cm[0,2]) + '],[' + str(cm[1,0]) + ',' + str(cm[1,1]) + ',' + str(cm[1,2]) + '],[' + str(cm[2,0]) + ',' + str(cm[2,1]) + ',' + str(cm[2,2]) + ']] \n')
    f.close()
    
    
    # days = int((len(y_actual)+len(y_to_predict))/1440)
    minutes = df['MINUTES']
    minutes_to_predict = np.linspace(minutes[y_start_predict], minutes[y_start_predict]+n_forecast-1, n_forecast)
    
    
    plt.figure(figsize=(36, 12))
    # plt.scatter(minutes[y_min:y_start_predict], y_actual, color='blue', label='Estados reales')
    # plt.scatter(df[(df['MINUTES']>=minutes[y_start_predict]) & (df['MINUTES']<minutes[y_start_predict]+n_forecast)]['MINUTES'], df[(df['MINUTES']>=minutes[y_start_predict]) & (df['MINUTES']<minutes[y_start_predict]+n_forecast)]['STATE_NUMBER'], color='green')
    plt.scatter(df[(df['MINUTES']<minutes[y_start_predict]+n_forecast)]['MINUTES'], df[(df['MINUTES']<minutes[y_start_predict]+n_forecast)]['STATE_NUMBER'], color='blue', label='Estados reales')
    plt.scatter(minutes_to_predict, Y_predict, color='red', label='Predicciones')
    # plt.xlim(minutes_to_predict[0] - 3*len(minutes_to_predict), minutes_to_predict[-1])
    # plt.legend()
    plt.savefig('Results/RNN_prediction_animal_' + animal + '_grupo_' + grupo + '.png')
    plt.show()