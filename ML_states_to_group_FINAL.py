import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from NN_FUNCTIONS import check, compare_values



# DATES

states = pd.read_csv('Dates/states_for_days_no_invalid.csv')

states_norm = pd.DataFrame()
states_norm['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
    

sc = StandardScaler()

y = states['GROUP_NUMBER']  # 1:33   2:53   3:27

len(states[states['GROUP_NUMBER']==1])
len(states[states['GROUP_NUMBER']==3])

states_norm_with_group = pd.DataFrame()
states_norm_with_group['GROUP'] = y #states['GROUP_NUMBER']
states_norm_with_group['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm_with_group['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]




# STATISTICS VALUES
    
corr = states_norm_with_group.corr()

plt.figure(figsize=(15,12))
ax = sns.heatmap(corr, annot=True, cmap = 'rocket_r')
# ax = sns.heatmap(corr, annot=True, cmap = 'Blues')
# plt.savefig('Results/correlaciones_states_binary.png')
plt.show()



# ------------------------------------------------- METHODS -----------------------------------------------------

# CLUSTER HIERARCHY (scipy.cluster.hierarchy)

X = sc.fit_transform(states_norm)

# Plot dendrogram

plt.figure(figsize=(16,8))
plt.title('Dendrograms')
dend = shc.dendrogram(shc.linkage(X, method = 'ward'))
plt.savefig('Results/Dendogram_for_days.png')
plt.show()


cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, tol=0.0001, random_state=0)

# kmeans.fit(X)
# labels = kmeans.predict(X)
# centroids  = kmeans.cluster_centers_ 

y_predict = cluster.fit_predict(X)
np.array([y])[0]


compare_values(y, y_predict, 1, 0) #GRUPO, CLUSTER
compare_values(y, y_predict, 1, 1)
compare_values(y, y_predict, 1, 2)
compare_values(y, y_predict, 1, 3)
compare_values(y, y_predict, 2, 0) # El cluster 0 clasifica bien el grupo 2
compare_values(y, y_predict, 2, 1) # El cluster 1 clasifica regular el grupo 2
compare_values(y, y_predict, 2, 2) # El cluster 2 clasifica bien el grupo 2
compare_values(y, y_predict, 2, 3)
compare_values(y, y_predict, 3, 0)
compare_values(y, y_predict, 3, 1)
compare_values(y, y_predict, 3, 2)
compare_values(y, y_predict, 3, 3)

# Logistic regression

states_for_logreg = pd.DataFrame()
states_for_logreg['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_for_logreg['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
    
states_for_logreg = states_for_logreg.drop(['S_10'], axis=1)
states_for_logreg['S_1_2'] = states_for_logreg['S_1'] * states_for_logreg['S_2']
states_for_logreg['S_1_3'] = states_for_logreg['S_1'] * states_for_logreg['S_3']
# states_for_logreg['S_2_3'] = states_for_logreg['S_2'] * states_for_logreg['S_3']
states_for_logreg['S_4_5'] = states_for_logreg['S_4'] * states_for_logreg['S_5']
states_for_logreg['S_1_11'] = states_for_logreg['S_1'] * states_for_logreg['S_11']
states_for_logreg['S_1_2_4'] = states_for_logreg['S_1_2'] * states_for_logreg['S_4']

# states_for_logreg['GROUP'] = y
# corr_logreg = states_for_logreg.corr()

# plt.figure(figsize=(15,12))
# ax = sns.heatmap(corr_logreg, annot=True, cmap = 'rocket_r')
# # ax = sns.heatmap(corr_logreg, annot=True, cmap = 'Blues')
# plt.savefig('Results/correlaciones_log_reg_states_binary.png')
# plt.show()



#  Support vector machine

# clf = svm.SVC(C=1.0, kernel='linear')
clf = svm.SVC(C=1.35, kernel='poly', degree=2, coef0=0)  # best result: degree = 2
# clf = svm.SVC(C=1.0, kernel='sigmoid')  # horrible
# clf = svm.SVC(C=1, kernel='rbf')  # BEST

scores = cross_val_score(clf, X, y, cv=5, scoring = 'f1_micro')
y_cv = cross_val_predict(clf, X, y, cv=5)

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
