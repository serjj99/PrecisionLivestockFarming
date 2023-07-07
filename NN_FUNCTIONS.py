import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as kr

def check(y, predict):
    y = np.array([y])[0]
    good = 0
    bad = 0
    for i in range(len(y)):
        if y[i] == predict[i]:
            good += 1
        else:
            bad += 1
    return good,bad



def compare_values(y, predict, value_y, value_predict):
    y = np.array([y])[0]
    good = 0
    bad = 0
    index = []
    for i in range(len(predict)):
        if value_predict == predict[i]:
            index.append(i)
            if y[i] == value_y:
                good += 1
            else:
                bad +=1
    return good,bad, np.array([index])



def binary(array):
    array_binary = np.array([])
    for i in array:
        if i>0.5:
            array_binary = np.append(array_binary, 1)
        else:
            array_binary = np.append(array_binary, 0)
    return array_binary



def predictions_models_cv(X, models):
    y_pred = np.zeros(len(X))
    for k in range(len(models)):
        y_pred += binary(models[k].predict(X))
    
    for i in range(len(y_pred)):
        if y_pred[i] < (len(models)/2):
            y_pred[i] = 0
        else:
            y_pred[i] = 1
            
    return y_pred



def separate_date(X, y, i, cv):
    if i == (cv-1):
        
        size_test = int(X.shape[0]/cv)
        
        x0 = int(i * size_test)

        X_test = X[x0:]
        X_train = X.drop(X.index[x0:], axis=0, inplace=False)
        
        y_test = y[x0:]
        y_train = y.drop(y.index[x0:], axis=0, inplace=False)
        
    else:
        size_test = int(X.shape[0]/cv)
        
        x0 = int(i * size_test)
        xn = int((i+1) * size_test)
        
        X_test = X[x0:xn]
        X_train = X.drop(X.index[x0:xn], axis=0, inplace=False)
        
        y_test = y[x0:xn]
        y_train = y.drop(y.index[x0:xn], axis=0, inplace=False)
    
    return X_train, X_test, y_train, y_test


def num_parametros(arquitectura):
    param = 0
    for i in range(1,len(arquitectura)):
        param += arquitectura[i-1]*arquitectura[i] + arquitectura[i]
    return param



                                    
def optimize_eprochs_neural_network(X_train, y_train, X_val, y_val, neuron, eprochs, seed_split = 0, name_txt = 'best_arquitecture.txt', seed_tf = 42, write = True, fun_act_hidden = 'relu', fun_act_output = 'softmax', LOSS = 'categorical_crossentropy', METRICS = 'categorical_accuracy', LR = 1e-2):
     
    np.random.seed(seed_tf)
    tf.random.set_seed(seed_tf)
    
    model = kr.Sequential()

    for n_neu in neuron[1:-1]:
        model.add(kr.layers.Dense(n_neu, activation=fun_act_hidden))
    model.add(kr.layers.Dense(neuron[-1], activation=fun_act_output))
    
    model.compile(loss=LOSS, optimizer=kr.optimizers.SGD(learning_rate = LR), metrics = METRICS)
    
    history = model.fit(X_train, y_train, epochs=eprochs, validation_data=(X_val, y_val))
    
    acc_train = history.history[METRICS]
    acc_val = history.history['val_' + METRICS]
    
    best_acc_train = 0
    best_acc_val = 0
    best_eproch = 0
    for i in range(len(acc_train)):
        if acc_train[i] >= acc_val[i]:
            if acc_val[i] > best_acc_val:
                best_acc_train = acc_train[i]
                best_acc_val = acc_val[i]
                best_eproch = int(i+1)
    
    if write: 
        f = open(name_txt, "a")
        f.write(str(int(seed_split)) + ' \t [')
        for n in neuron[:-1]:
            f.write(str(int(n)) + ",")
    
        f.write(str(int(neuron[-1])) + '] \t' + str(best_eproch) + '\t' + str(round(best_acc_train, 4)) + '\t' + str(round(best_acc_val, 4)) + '\n')
        f.close()
        
    return best_eproch, best_acc_train, best_acc_val #, acc_train, acc_val



def optimize_complete_neural_network(X, y, val_size = 0.25, eprochs = 1000, layers = [3,4], seed_split = 0, name_txt = 'best_arquitecture.txt', seed_tf = 42, write = True, output_layer = 3, fun_act_hidden = 'relu', fun_act_output = 'softmax', LOSS = 'categorical_crossentropy', METRICS = 'categorical_accuracy', LR = 1e-2):
    if write:
        f = open(name_txt, "w")
        f.write('SEED \t ARQUITECTURE \t EPROCH \t ACC_TRAIN \t ACC_VAL \n')
        f.close()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_size, random_state = seed_split)
    
    sc = StandardScaler() 
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    
    # y_train_ori = y_train
    # y_val_ori = y_val
    
    
    p = X.shape[1]
    max_param = int(X_train.shape[0]/3)
    output_layer = int(output_layer)
    
    if output_layer == 1:
        y_train = y_train-min(y)
        y_val = y_val-min(y)
    else:
        y_train = kr.utils.to_categorical(y_train-min(y))
        y_val = kr.utils.to_categorical(y_val-min(y))
    
    index3=0
    i = 20
    while num_parametros([p,i,output_layer]) < max_param:
        i += 5
        index3 += 1
            
    index4 = 0
    i = 3
    while i <= 30:
        j = 3
        while j <= 30:
            if num_parametros([p,i,j,output_layer]) < max_param:
                j += 3
                index4 += 1
            else:
                j += 3
        i += 3
        
    index5 = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 24:
            k = 3
            while k <= 24:
                if num_parametros([p,i,j,k,output_layer]) < max_param:
                    k += 3
                    index5 += 1
                else:
                    k += 3
            j += 3
        i += 3
        
    index6 = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 21:
            k = 3
            while k <= 18:
                l = 3
                while l<= 15:
                    if num_parametros([p,i,j,k,l,output_layer]) < max_param:
                        l += 3
                        index6 += 1
                    else:
                        l += 3
                k += 3
            j += 3
        i += 3
        
    neurons3 = np.zeros((int(index3), 3))
    neurons4 = np.zeros((int(index4), 4))
    neurons5 = np.zeros((int(index5), 5))
    neurons6 = np.zeros((int(index6), 6))
    
    neurons3[:,0] = p
    neurons3[:,-1] = output_layer
    neurons4[:,0] = p
    neurons4[:,-1] = output_layer
    neurons5[:,0] = p
    neurons5[:,-1] = output_layer
    neurons6[:,0] = p
    neurons6[:,-1] = output_layer
    
    
    index = 0
    i = 20
    while num_parametros([p,i,output_layer]) < max_param:
        neurons3[index, 1] = i
        i += 5
        index += 1
        
        
    index = 0
    i = 3
    while i <= 30:
        j = 3
        while j <= 30:
            if num_parametros([p,i,j,output_layer]) < max_param:
                neurons4[index, 1] = i
                neurons4[index, 2] = j
                j += 3
                index += 1
            else:
                j += 3
        i += 3
        
    index = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 24:
            k = 3
            while k <= 24:
                if num_parametros([p,i,j,k,output_layer]) < max_param:
                    neurons5[index, 1] = i
                    neurons5[index, 2] = j
                    neurons5[index, 3] = k
                    k += 3
                    index += 1
                else:
                    k += 3
            j += 3
        i += 3
        
    index = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 21:
            k = 3
            while k <= 18:
                l = 3
                while l<= 15:
                    if num_parametros([p,i,j,k,l,output_layer]) < max_param:
                        neurons6[index, 1] = i
                        neurons6[index, 2] = j
                        neurons6[index, 3] = k
                        neurons6[index, 4] = l
                        l += 3
                        index += 1
                    else:
                        l += 3
                k += 3
            j += 3
        i += 3
             
    
    BEST_ARQUITECTURE = 0
    BEST_ACC_TRAIN = 0
    BEST_ACC_VAL = 0
    BEST_EPROCH = 0
    
    if 3 in layers:
        for neuron in neurons3:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_neural_network(X_train, y_train, X_val, y_val, neuron, eprochs, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
    
    if 4 in layers:
        for neuron in neurons4:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_neural_network(X_train, y_train, X_val, y_val, neuron, eprochs, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
    
    if 5 in layers:
        for neuron in neurons5:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_neural_network(X_train, y_train, X_val, y_val, neuron, eprochs, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
        
    if 6 in layers:
        for neuron in neurons6:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_neural_network(X_train, y_train, X_val, y_val, neuron, eprochs, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
            
    if write:           
        f = open(name_txt, "a")
        f.write('\n SEED \t BEST_ARQUITECTURE \t BEST_EPROCH \t BEST_ACC_TRAIN \t BEST_ACC_VAL \n')
        
        f.write(str(int(seed_split)) + ' \t [')
        
        for n in BEST_ARQUITECTURE[:-1]:
            f.write(str(int(n)) + ",")
            
        f.write(str(int(BEST_ARQUITECTURE[-1])) + '] \t' + str(int(BEST_EPROCH)) + '\t' + str(round(BEST_ACC_TRAIN, 6)) + '\t' + str(round(BEST_ACC_VAL, 6)) + '\n \n \n')
        f.close()
    
    return BEST_ARQUITECTURE, BEST_ACC_TRAIN, BEST_ACC_VAL, BEST_EPROCH









def cross_validation_neural_network(X, y, neurons, eprochs, all_history = False, plot = False, cv = 5, seed_tf = 42, fun_act_hidden = 'relu', fun_act_output = 'softmax', LOSS = 'categorical_crossentropy', METRICS = 'categorical_accuracy', LR = 1e-2):
    if all_history:
        sc = StandardScaler()
        acc_train = []
        acc_test = []
        
        for i in range(cv):
            X_train, X_test, y_train, y_test = separate_date(X, y, i, cv)
            
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    
            if neurons[-1] == 1:
                y_train = y_train-min(y)
                y_test = y_test-min(y)
            else:
                y_train = kr.utils.to_categorical(y_train-min(y))
                y_test = kr.utils.to_categorical(y_test-min(y))
            
            np.random.seed(seed_tf)
            tf.random.set_seed(seed_tf)
            
            model = kr.Sequential()
        
            for n_neu in neurons[1:-1]:
                model.add(kr.layers.Dense(n_neu, activation=fun_act_hidden))
            model.add(kr.layers.Dense(neurons[-1], activation=fun_act_output))
            
            model.compile(loss=LOSS, optimizer=kr.optimizers.SGD(learning_rate = LR), metrics = METRICS)
            
            history = model.fit(X_train, y_train, epochs=eprochs, validation_data=(X_test, y_test))
            
            acc_train.append(history.history[METRICS])
            acc_test.append(history.history['val_' + METRICS])
            
        acc_train_cv = np.zeros(eprochs)
        acc_test_cv = np.zeros(eprochs)
        
        for j in range(eprochs):
            for k in range(cv):
                acc_train_cv[j] += acc_train[k][j]
                acc_test_cv[j] += acc_test[k][j]
            acc_train_cv[j] = acc_train_cv[j] / cv
            acc_test_cv[j] = acc_test_cv[j] / cv
        
        if plot:
            plt.figure(figsize=(8,8))
            plt.title('ACCURACY {}'.format(neurons))
            plt.plot(acc_train_cv, color='blue', label='Train acc')
            plt.plot(acc_test_cv, color='red', label='Test acc')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.show()
            
        return acc_train_cv, acc_test_cv
    
    else:
        sc = StandardScaler()
        acc_train_cv_all = np.array([])
        acc_test_cv_all = np.array([])
        models_cv = []
        
        for i in range(cv):
            X_train, X_test, y_train, y_test = separate_date(X, y, i, cv)
            
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    
            if neurons[-1] == 1:
                y_train = y_train-min(y)
                y_test = y_test-min(y)
            else:
                y_train = kr.utils.to_categorical(y_train-min(y))
                y_test = kr.utils.to_categorical(y_test-min(y))
            
            np.random.seed(seed_tf)
            tf.random.set_seed(seed_tf)
            
            model = kr.Sequential()
        
            for n_neu in neurons[1:-1]:
                model.add(kr.layers.Dense(n_neu, activation=fun_act_hidden))
            model.add(kr.layers.Dense(neurons[-1], activation=fun_act_output))
            
            model.compile(loss=LOSS, optimizer=kr.optimizers.SGD(learning_rate = LR), metrics = METRICS)
            
            history = model.fit(X_train, y_train, epochs=eprochs, validation_data=(X_test, y_test))
            
            models_cv.append(model) 
            
            acc_train_cv_all = np.append(acc_train_cv_all, history.history[METRICS][-1])
            acc_test_cv_all = np.append(acc_test_cv_all, history.history['val_' + METRICS][-1])
        
        acc_train_cv = acc_train_cv_all.mean()
        acc_test_cv = acc_test_cv_all.mean()
        acc_train_cv_std = acc_train_cv_all.std()
        acc_test_cv_std = acc_test_cv_all.std()
        
        return acc_train_cv_all, acc_test_cv_all, acc_train_cv, acc_test_cv, acc_train_cv_std, acc_test_cv_std, models_cv







def cross_validation_neural_network_models(X, y, neurons, eprochs, all_history = False, plot = False, cv = 5, seed_tf = 42, fun_act_hidden = 'relu', fun_act_output = 'softmax', LOSS = 'categorical_crossentropy', METRICS = 'categorical_accuracy', LR = 1e-2):
    if all_history:
        sc = StandardScaler()
        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = []
        
        for i in range(cv):
            X_train, X_test, y_train, y_test = separate_date(X, y, i, cv)
            
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    
            if neurons[-1] == 1:
                y_train = y_train-min(y)
                y_test = y_test-min(y)
            else:
                y_train = kr.utils.to_categorical(y_train-min(y))
                y_test = kr.utils.to_categorical(y_test-min(y))
            
            np.random.seed(seed_tf)
            tf.random.set_seed(seed_tf)
            
            model = kr.Sequential()
        
            for n_neu in neurons[1:-1]:
                model.add(kr.layers.Dense(n_neu, activation=fun_act_hidden))
            model.add(kr.layers.Dense(neurons[-1], activation=fun_act_output))
            
            model.compile(loss=LOSS, optimizer=kr.optimizers.SGD(learning_rate = LR), metrics = METRICS)
            
            history = model.fit(X_train, y_train, epochs=eprochs, validation_data=(X_test, y_test))
            
            acc_train.append(history.history[METRICS])
            acc_test.append(history.history['val_' + METRICS])
            loss_train.append(history.history['loss'])
            loss_test.append(history.history['val_loss'])
            
        acc_train_cv = np.zeros(eprochs)
        acc_test_cv = np.zeros(eprochs)
        loss_train_cv = np.zeros(eprochs)
        loss_test_cv = np.zeros(eprochs)
        
        for j in range(eprochs):
            for k in range(cv):
                acc_train_cv[j] += acc_train[k][j]
                acc_test_cv[j] += acc_test[k][j]
                loss_train_cv[j] += loss_train[k][j]
                loss_test_cv[j] += loss_test[k][j]
            acc_train_cv[j] = acc_train_cv[j] / cv
            acc_test_cv[j] = acc_test_cv[j] / cv
            loss_train_cv[j] = loss_train_cv[j] / cv
            loss_test_cv[j] = loss_test_cv[j] / cv
        
        if plot:
            plt.figure(figsize=(6,6))
            plt.title('LOSS')
            plt.plot(loss_train_cv, color='blue', label='Train loss')
            plt.plot(loss_test_cv, color='red', label='Test loss')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.show()
            
            plt.figure(figsize=(6,6))
            plt.title('ACCURACY')
            plt.plot(acc_train_cv, color='blue', label='Train acc')
            plt.plot(acc_test_cv, color='red', label='Test acc')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.show()
            
        return acc_train_cv, acc_test_cv
    
    else:
        sc = StandardScaler()
        acc_train_cv_all = np.array([])
        acc_test_cv_all = np.array([])
        models_cv = []
        
        for i in range(cv):
            X_train, X_test, y_train, y_test = separate_date(X, y, i, cv)
            
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    
            if neurons[-1] == 1:
                y_train = y_train-min(y)
                y_test = y_test-min(y)
            else:
                y_train = kr.utils.to_categorical(y_train-min(y))
                y_test = kr.utils.to_categorical(y_test-min(y))
            
            np.random.seed(seed_tf)
            tf.random.set_seed(seed_tf)
            
            model = kr.Sequential()
        
            for n_neu in neurons[1:-1]:
                model.add(kr.layers.Dense(n_neu, activation=fun_act_hidden))
            model.add(kr.layers.Dense(neurons[-1], activation=fun_act_output))
            
            model.compile(loss=LOSS, optimizer=kr.optimizers.SGD(learning_rate = LR), metrics = METRICS)
            
            history = model.fit(X_train, y_train, epochs=eprochs, validation_data=(X_test, y_test))
            
            models_cv.append(model) 
            
            acc_train_cv_all = np.append(acc_train_cv_all, history.history[METRICS][-1])
            acc_test_cv_all = np.append(acc_test_cv_all, history.history['val_' + METRICS][-1])
        
        acc_train_cv = acc_train_cv_all.mean()
        acc_test_cv = acc_test_cv_all.mean()
        acc_train_cv_std = acc_train_cv_all.std()
        acc_test_cv_std = acc_test_cv_all.std()
        
        # return acc_train_cv_all, acc_test_cv_all, acc_train_cv, acc_test_cv, acc_train_cv_std, acc_test_cv_std, models_cv
        return models_cv, acc_test_cv_all










def optimize_eprochs_cv_neural_network(X, y, neuron, eprochs, cv, seed_split = 0, name_txt = 'best_arquitecture_cv.txt', seed_tf = 42, write = True, fun_act_hidden = 'relu', fun_act_output = 'softmax', LOSS = 'categorical_crossentropy', METRICS = 'categorical_accuracy', LR = 1e-2):

    sc = StandardScaler()
    acc_train = []
    acc_test = []
    
    for i in range(cv):
        X_train, X_test, y_train, y_test = separate_date(X, y, i, cv)
        
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if neuron[-1] == 1:
            y_train = y_train-min(y)
            y_test = y_test-min(y)
        else:
            y_train = kr.utils.to_categorical(y_train-min(y))
            y_test = kr.utils.to_categorical(y_test-min(y))
        
        np.random.seed(seed_tf)
        tf.random.set_seed(seed_tf)
        
        model = kr.Sequential()
    
        for n_neu in neuron[1:-1]:
            model.add(kr.layers.Dense(n_neu, activation=fun_act_hidden))
        model.add(kr.layers.Dense(neuron[-1], activation=fun_act_output))
        
        model.compile(loss=LOSS, optimizer=kr.optimizers.SGD(learning_rate = LR), metrics = METRICS)
        
        history = model.fit(X_train, y_train, epochs=eprochs, validation_data=(X_test, y_test))
        
        acc_train.append(history.history[METRICS])
        acc_test.append(history.history['val_' + METRICS])
    
    best_acc_train = 0
    best_acc_test = 0
    best_std_train = 0
    best_std_test = 0
    best_eproch = 0
    
    
    for j in range(eprochs):
        acc_train_cv = 0
        acc_test_cv = 0
        std_train_cv_list = []
        std_test_cv_list = []
        
        for k in range(cv):
            acc_train_cv += acc_train[k][j]
            acc_test_cv += acc_test[k][j]
            std_train_cv_list.append(acc_train[k][j])
            std_test_cv_list.append(acc_test[k][j])
        acc_train_cv = acc_train_cv / cv
        acc_test_cv = acc_test_cv / cv
        std_train_cv = statistics.stdev(std_train_cv_list)
        std_test_cv = statistics.stdev(std_test_cv_list)
        
        if acc_test_cv > best_acc_test:
            best_acc_train = acc_train_cv
            best_acc_test = acc_test_cv
            best_std_train = std_train_cv
            best_std_test = std_test_cv
            best_eproch = int(j+1)
            
    
    if write: 
        f = open(name_txt, "a")
        f.write(str(int(cv)) + ' \t [')
        for n in neuron[:-1]:
            f.write(str(int(n)) + ",")
    
        f.write(str(int(neuron[-1])) + '] \t' + str(best_eproch) + '\t' + str(round(best_acc_train, 4)) + '\t' + str(round(best_acc_test, 4)) + '\t' + str(round(best_std_train, 4)) + '\t' + str(round(best_std_test, 4))  + '\n')
        f.close()
        
    return best_eproch, best_acc_train, best_acc_test




def optimize_complete_cv_neural_network(X, y, cv = 5, eprochs = 1000, layers = [3,4], seed_split = 0, name_txt = 'best_arquitecture_cv.txt', seed_tf = 42, write = True, output_layer = 3, fun_act_hidden = 'relu', fun_act_output = 'softmax', LOSS = 'categorical_crossentropy', METRICS = 'categorical_accuracy', LR = 1e-2):
    if write:
        f = open(name_txt, "w")
        f.write('CV \t ARQUITECTURE \t EPROCH \t ACC_TRAIN \t ACC_TEST \t STD_TRAIN \t STD_TEST \n')
        f.close()

    
    p = X.shape[1]
    max_param = int((X.shape[0]/3) * (1-(1/cv)))
    output_layer = int(output_layer)
    
    
    index3=0
    i = 3
    while (num_parametros([p,i,output_layer]) < max_param) and (i<=30):
        i += 3
        index3 += 1
            
    index4 = 0
    i = 3
    while i <= 30:
        j = 3
        while j <= 30:
            if num_parametros([p,i,j,output_layer]) < max_param:
                j += 3
                index4 += 1
            else:
                j += 3
        i += 3
        
    index5 = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 24:
            k = 3
            while k <= 24:
                if num_parametros([p,i,j,k,output_layer]) < max_param:
                    k += 3
                    index5 += 1
                else:
                    k += 3
            j += 3
        i += 3
        
    index6 = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 21:
            k = 3
            while k <= 18:
                l = 3
                while l<= 15:
                    if num_parametros([p,i,j,k,l,output_layer]) < max_param:
                        l += 3
                        index6 += 1
                    else:
                        l += 3
                k += 3
            j += 3
        i += 3
        
    neurons3 = np.zeros((int(index3), 3))
    neurons4 = np.zeros((int(index4), 4))
    neurons5 = np.zeros((int(index5), 5))
    neurons6 = np.zeros((int(index6), 6))
    
    neurons3[:,0] = p
    neurons3[:,-1] = output_layer
    neurons4[:,0] = p
    neurons4[:,-1] = output_layer
    neurons5[:,0] = p
    neurons5[:,-1] = output_layer
    neurons6[:,0] = p
    neurons6[:,-1] = output_layer
    
    
    index = 0
    i = 3
    while (num_parametros([p,i,output_layer]) < max_param) and (i<=30):
        neurons3[index, 1] = i
        i += 3
        index += 1
        
        
    index = 0
    i = 3
    while i <= 30:
        j = 3
        while j <= 30:
            if num_parametros([p,i,j,output_layer]) < max_param:
                neurons4[index, 1] = i
                neurons4[index, 2] = j
                j += 3
                index += 1
            else:
                j += 3
        i += 3
        
    index = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 24:
            k = 3
            while k <= 24:
                if num_parametros([p,i,j,k,output_layer]) < max_param:
                    neurons5[index, 1] = i
                    neurons5[index, 2] = j
                    neurons5[index, 3] = k
                    k += 3
                    index += 1
                else:
                    k += 3
            j += 3
        i += 3
        
    index = 0
    i = 3
    while i <= 24:
        j = 3
        while j <= 21:
            k = 3
            while k <= 18:
                l = 3
                while l<= 15:
                    if num_parametros([p,i,j,k,l,output_layer]) < max_param:
                        neurons6[index, 1] = i
                        neurons6[index, 2] = j
                        neurons6[index, 3] = k
                        neurons6[index, 4] = l
                        l += 3
                        index += 1
                    else:
                        l += 3
                k += 3
            j += 3
        i += 3
        
            
    
    BEST_ARQUITECTURE = 0
    BEST_ACC_TRAIN = 0
    BEST_ACC_VAL = 0
    BEST_EPROCH = 0
    
    if 2 in layers:
        for neuron in [[p,output_layer]]:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_cv_neural_network(X, y, neuron, eprochs, cv, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
                
    if 3 in layers:
        for neuron in neurons3:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_cv_neural_network(X, y, neuron, eprochs, cv, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
    
    if 4 in layers:
        for neuron in neurons4[32:]:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_cv_neural_network(X, y, neuron, eprochs, cv, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
    
    if 5 in layers:
        for neuron in neurons5:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_cv_neural_network(X, y, neuron, eprochs, cv, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
        
    if 6 in layers:
        for neuron in neurons6:
            best_eproch, best_acc_train, best_acc_val = optimize_eprochs_cv_neural_network(X, y, neuron, eprochs, cv, seed_split, name_txt, seed_tf, write, fun_act_hidden, fun_act_output, LOSS, METRICS, LR)
            if best_acc_val > BEST_ACC_VAL:
                BEST_ARQUITECTURE = neuron
                BEST_ACC_TRAIN = best_acc_train
                BEST_ACC_VAL = best_acc_val
                BEST_EPROCH = best_eproch
            
    if write:           
        f = open(name_txt, "a")
        f.write('\n CV \t BEST_ARQUITECTURE \t BEST_EPROCH \t BEST_ACC_TRAIN \t BEST_ACC_VAL \n')
        
        f.write(str(int(cv)) + ' \t [')
        
        for n in BEST_ARQUITECTURE[:-1]:
            f.write(str(int(n)) + ",")
            
        f.write(str(int(BEST_ARQUITECTURE[-1])) + '] \t' + str(int(BEST_EPROCH)) + '\t' + str(round(BEST_ACC_TRAIN, 6)) + '\t' + str(round(BEST_ACC_VAL, 6)) + '\n \n \n')
        f.close()
    
    return BEST_ARQUITECTURE, BEST_ACC_TRAIN, BEST_ACC_VAL, BEST_EPROCH