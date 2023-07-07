import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from NN_FUNCTIONS import check, compare_values



# DATES

states = pd.read_csv('Dates/states_for_days_no_invalid.csv')

states_norm = pd.DataFrame()
states_norm['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
    

sc = StandardScaler()

y = states['GROUP_NUMBER']  # 1:33   2:53   3:27
y = y.replace([3], 1)

X = sc.fit_transform(states_norm)


states_norm_with_group = pd.DataFrame()
states_norm_with_group['GROUP'] = y #states['GROUP_NUMBER']
states_norm_with_group['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm_with_group['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]



# STATISTICS VALUES
    
# corr = states_norm_with_group.corr()

# plt.figure(figsize=(15,12))
# ax = sns.heatmap(corr, annot=True, cmap = 'rocket_r')
# # ax = sns.heatmap(corr, annot=True, cmap = 'Blues')
# # plt.savefig('Results/correlaciones_states_binary.png')
# plt.show()



# ------------------------------------------------- METHODS -----------------------------------------------------

# AGGLOMERATIVE CLUSTERING

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit(X)

y_pred_cluster = cluster.fit_predict(X)
for i in range(len(y_pred_cluster)):
    if y_pred_cluster[i] == 0:
        y_pred_cluster[i] = 2

print('CM cluster =', confusion_matrix(y, y_pred_cluster))
print('Accuracy cluster:', f1_score(y, y_pred_cluster, average='micro') *100, '%\n')
        

# K MEANS

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

y_pred_kmeans = kmeans.predict(X)
for i in range(len(y_pred_kmeans)):
    if y_pred_kmeans[i] == 0:
        y_pred_kmeans[i] = 2
                
print('CM kmean =', confusion_matrix(y, y_pred_kmeans))
print('Accuracy kmean:', f1_score(y, y_pred_kmeans, average='micro') *100, '%\n')



#  Support vector machine

# clf = svm.SVC(C=1.0, kernel='linear')
# clf = svm.SVC(C=1.0, kernel='poly', degree=2)  # best result: degree = 2
# clf = svm.SVC(C=1.0, kernel='sigmoid')  # horrible
clf = svm.SVC(C=0.7, kernel='rbf')  # BEST

scores = cross_val_score(clf, X, y, cv=5, scoring = 'f1_micro')
y_cv = cross_val_predict(clf, X, y, cv=5)

cm_cv = confusion_matrix(y, y_cv)

print('CM cv svm = ', cm_cv)
print('Acc cv svm =', scores.mean()*100, '%')
print('Std cv svm =', scores.std(), '\n')


# Prueba con el state 10

def predict_with_state_10(X):
    pred = np.array([])
    for i in range(X.shape[0]):
        if X['S_10'][i] == 0:
            pred = np.append(pred, 2)
        else:
            pred = np.append(pred, 1)
    return pred

def check_search_errors(y, predict):
    y = np.array([y])[0]
    good = 0
    bad = 0
    error_1 = []
    error_2 = []
    for i in range(len(y)):
        if y[i] == predict[i]:
            good += 1
        else:
            bad += 1
            if y[i] == 1:
                error_1.append(i)
            else:
                error_2.append(i)
    acc = good / (good + bad) *100
    return good, bad, acc, error_1, error_2
          

pred_with_state_10 = predict_with_state_10(states_norm)

good, bad, acc, error_1, error_2 = check_search_errors(y, pred_with_state_10)

confusion_matrix(y, pred_with_state_10)

# Aquí podemos ver que hay destetadas con el state 10 != 0 (lactando) y no destetadas con el state 10 = 0 (no lactan un dia)


# Logistic regression

states_for_logreg = pd.DataFrame()
states_for_logreg['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_for_logreg['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
    
states_for_logreg = states_for_logreg.drop(['S_10'], axis=1)
states_for_logreg['S_1_2'] = states_for_logreg['S_1'] * states_for_logreg['S_2']
states_for_logreg['S_1_3'] = states_for_logreg['S_1'] * states_for_logreg['S_3']
states_for_logreg['S_4_5'] = states_for_logreg['S_4'] * states_for_logreg['S_5']
states_for_logreg['S_1_11'] = states_for_logreg['S_1'] * states_for_logreg['S_11']
states_for_logreg['S_1_2_4'] = states_for_logreg['S_1_2'] * states_for_logreg['S_4']

states_for_logreg['GROUP'] = y
corr_logreg = states_for_logreg.corr()

plt.figure(figsize=(15,12))
ax = sns.heatmap(corr_logreg, annot=True, cmap = 'rocket_r')
# ax = sns.heatmap(corr_logreg, annot=True, cmap = 'Blues')
plt.savefig('Results/correlaciones_log_reg_states_binary.png')
plt.show()

classifier = LogisticRegression(penalty = 'l2', tol = 1e-5, max_iter = int(1e4), random_state = 0)
scores = cross_val_score(classifier, states_for_logreg, y, cv=5, scoring = 'f1_micro')
y_cv = cross_val_predict(classifier, states_for_logreg, y, cv=5)

cm_cv = confusion_matrix(y, y_cv)

print('CM cv svm = ', cm_cv)
print('Acc cv svm =', scores.mean()*100, '%')
print('Std cv svm =', scores.std(), '\n')


# VALIDATION and TEST

# X_train, X_test_val, y_train, y_test_val = train_test_split(states_for_logreg, y, test_size = 0.3, random_state = 0)
# X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size = 0.5, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(states_for_logreg, y, test_size = 0.25, random_state = 0)

X_train = sc.fit_transform(X_train)
# X_val = sc.transform(X_val)
X_test = sc.transform(X_test)


classifier = LogisticRegression(penalty = 'l2', tol = 1e-5, max_iter = int(1e4), random_state = 0)
classifier.fit(X_train, y_train)



print('Constante = ', classifier.intercept_[0])
print('Coeficientes: \n',classifier.coef_)


y_pred_train = classifier.predict(X_train)
# y_pred_val = classifier.predict(X_val)
y_pred_test = classifier.predict(X_test)


cmTrain = confusion_matrix(y_train, y_pred_train)
# cmVal = confusion_matrix(y_val, y_pred_val)
cmTest = confusion_matrix(y_test, y_pred_test)
cmAll = cmTrain + cmTest # + cmVal





print('CM Train =', cmTrain)
print('Good:', sum(np.diag(cmTrain)))
print('Error:', sum(sum(cmTrain)) - sum(np.diag(cmTrain)))
print('Accuracy log reg:', f1_score(y_train, y_pred_train, average='micro') *100, '%\n')

# print('CM Val =', cmVal)
# print('Good:', sum(np.diag(cmVal)))
# print('Error:', sum(sum(cmVal)) - sum(np.diag(cmVal)))
# print('Accuracy log reg:', f1_score(y_val, y_pred_val, average='micro') *100, '%\n')

print('CM Test =', cmTest)
print('Good:', sum(np.diag(cmTest)))
print('Error:', sum(sum(cmTest)) - sum(np.diag(cmTest)))
print('Accuracy log reg:', f1_score(y_test, y_pred_test, average='micro') *100, '%\n')

print('CM All =', cmAll)
print('Good:', sum(np.diag(cmAll)))
print('Error:', sum(sum(cmAll)) - sum(np.diag(cmAll)))
print('Accuracy all log reg:', sum(np.diag(cmAll)) / sum(sum(cmAll)) *100 , '%\n')




# CROSS VALIDATION

X = sc.fit_transform(states_for_logreg)

scores = cross_val_score(classifier, X, y, cv=5, scoring = 'f1_micro')
y_cv = cross_val_predict(classifier, X, y, cv=5)

cm_cv = confusion_matrix(y, y_cv)

print('CM_cv = ', cm_cv)
print('Acc cv =', scores.mean()*100, '%')
print('Std cv =', scores.std())
