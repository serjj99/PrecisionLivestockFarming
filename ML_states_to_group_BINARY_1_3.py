import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from NN_FUNCTIONS import check, compare_values



# DATES

states = pd.read_csv('Dates/states_for_days_no_invalid_1_3.csv')

states_norm = pd.DataFrame()
states_norm['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_norm['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
    

sc = StandardScaler()

y = states['GROUP_NUMBER']  # 1:33   2:53   3:27


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
# plt.savefig('Results/correlaciones_states_binary_1_3.png')
# plt.show()



# ------------------------------------------------- METHODS -----------------------------------------------------

# Logistic regression

states_for_logreg = pd.DataFrame()
states_for_logreg['S_1'] = states['STATE_NUMBER_NORM_1']
for i in range(3,13):
    states_for_logreg['S_{}'.format(i-1)] = states['STATE_NUMBER_NORM_{}'.format(i-1)]
    
states_for_logreg = states_for_logreg.drop(['S_6'], axis=1)
states_for_logreg = states_for_logreg.drop(['S_7'], axis=1)
states_for_logreg['S_1_2'] = states_for_logreg['S_1'] * states_for_logreg['S_2']
states_for_logreg['S_1_3'] = states_for_logreg['S_1'] * states_for_logreg['S_3']
states_for_logreg['S_4_5'] = states_for_logreg['S_4'] * states_for_logreg['S_5']
states_for_logreg['S_1_11'] = states_for_logreg['S_1'] * states_for_logreg['S_11']
states_for_logreg['S_1_2_4'] = states_for_logreg['S_1_2'] * states_for_logreg['S_4']
states_for_logreg['S_10_11'] = states_for_logreg['S_10'] * states_for_logreg['S_11']
states_for_logreg['S_1_4'] = states_for_logreg['S_1'] * states_for_logreg['S_4']

states_for_logreg['GROUP'] = y
corr_logreg = states_for_logreg.corr()

plt.figure(figsize=(15,12))
ax = sns.heatmap(corr_logreg, annot=True, cmap = 'rocket_r')
# ax = sns.heatmap(corr_logreg, annot=True, cmap = 'Blues')
# plt.savefig('Results/correlaciones_log_reg_states_binary.png')
plt.show()



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