# BIA-656
# HW 2
# Author: Vivek John D Martins
# SID: 10406763


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:\Users\Vivek\.spyder2\directMarketing.csv")

##Exploratory Data Analysis
df.head()

#checking for Null Values
print 'checking for Null Values'
list=df.columns
for col in list:
    print pd.value_counts(df[col].isnull())

#checking attribute types
df.dtypes

#creating dummy values
dummies1 = pd.get_dummies(df.saleSizeCode)
dummies2 = pd.get_dummies(df.starCustomer)

#dummies1.head()
#dummies2.head()

#concatenating 
dummies= pd.concat([dummies1, dummies2], axis=1)
df= pd.concat([df, dummies], axis=1)

#df.head()

#splitting the data into Training and Test
is_test = np.random.uniform(0, 1, len(df)) > 0.75
train = df[is_test==False]
test = df[is_test==True]

dumdrop=['starCustomer','saleSizeCode']

train = train.drop(dumdrop, 1)
test = test.drop(dumdrop, 1)

#train.head()

#features required for algorithms
feature_names= train.columns.tolist()
feature_names.remove('class')

from sklearn import tree
clf = tree.DecisionTreeClassifier()

from sklearn.svm import SVC
svc= SVC()

from sklearn.linear_model import LogisticRegression
lreg= LogisticRegression()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()




#training 
clf.fit(train[feature_names], train['class'])
svc.fit(train[feature_names], train['class'])
lreg.fit(train[feature_names], train['class'])
gnb.fit(train[feature_names], train['class'])


#Predict
clf.predict(test[feature_names])
svc.predict(test[feature_names])
lreg.predict(test[feature_names])
gnb.predict(test[feature_names])

#---------------------------------------------------------------------------------------------------------------------------------------------
#__Logistic Regression__

#crosstable
print '\n'
print pd.crosstab(test['class'], lreg.predict(test[feature_names]),rownames=['Actual'], colnames=['Predicted'], margins=True)

#error rate
from sklearn.metrics import accuracy_score
accuracy_lreg = accuracy_score(test['class'], lreg.predict(test[feature_names]), normalize=False) / float(test['class'].size)
error_rate_lreg = 1 - accuracy_lreg
print '\nLreg error rate',error_rate_lreg

#Area Under the ROC curve
from sklearn.metrics import roc_auc_score
auc_lreg=roc_auc_score(test['class'], lreg.predict(test[feature_names]))

# Determine the false positive and true positive rates
from sklearn.metrics import roc_curve
lreg_fpr, lreg_tpr, lreg_threshold = roc_curve(test['class'], lreg.predict(test[feature_names]))

# Determine precision recall
from sklearn.metrics import precision_recall_curve
precision_lreg, recall_lreg, _ = precision_recall_curve(test['class'], lreg.predict(test[feature_names]))

#Area Under the precision recall curve
from sklearn.metrics import average_precision_score
prc_auc_lreg=average_precision_score(test['class'], lreg.predict(test[feature_names]))

#---------------------------------------------------------------------------------------------------------------------------------------------
#__CART__

#crosstable
print '\n'
print pd.crosstab(test['class'], clf.predict(test[feature_names]),rownames=['Actual'], colnames=['Predicted'], margins=True)

#error rate
accuracy_clf = accuracy_score(test['class'], clf.predict(test[feature_names]), normalize=False) / float(test['class'].size)
error_rate_clf = 1 - accuracy_clf
print '\nCart error rate',error_rate_clf

#Area Under the ROC curve
auc_clf=roc_auc_score(test['class'], clf.predict(test[feature_names]))

# Determine the false positive and true positive rates
clf_fpr, clf_tpr, clf_threshold = roc_curve(test['class'], clf.predict(test[feature_names]))

# Determine precision recall
precision_clf, recall_clf, _ = precision_recall_curve(test['class'], clf.predict(test[feature_names]))

#Area Under the precision recall curve
prc_auc_clf=average_precision_score(test['class'], clf.predict(test[feature_names]))

#---------------------------------------------------------------------------------------------------------------------------------------------
#__SVM__

#crosstable
print '\n'
print pd.crosstab(test['class'], svc.predict(test[feature_names]),rownames=['Actual'], colnames=['Predicted'], margins=True)

#error rate
accuracy_svc = accuracy_score(test['class'], svc.predict(test[feature_names]), normalize=False) / float(test['class'].size)
error_rate_svc = 1 - accuracy_svc
print '\nSVM error rate',error_rate_svc

#Area Under the ROC curve
auc_svc=roc_auc_score(test['class'], svc.predict(test[feature_names]))

# Determine the false positive and true positive rates
svc_fpr, svc_tpr, svc_threshold = roc_curve(test['class'], svc.predict(test[feature_names]))

# Determine precision recall
precision_svc, recall_svc, _ = precision_recall_curve(test['class'], svc.predict(test[feature_names]))

#Area Under the precision recall curve
prc_auc_svc=average_precision_score(test['class'], svc.predict(test[feature_names]))

#---------------------------------------------------------------------------------------------------------------------------------------------
#__Naive Bayes__

#crosstable
print '\n'
print pd.crosstab(test['class'], gnb.predict(test[feature_names]),rownames=['Actual'], colnames=['Predicted'], margins=True)

#error rate
accuracy_gnb = accuracy_score(test['class'], gnb.predict(test[feature_names]), normalize=False) / float(test['class'].size)
error_rate_gnb = 1 - accuracy_gnb
print '\nNB error rate',error_rate_gnb

#Area Under the ROC curve
auc_gnb=roc_auc_score(test['class'], gnb.predict(test[feature_names]))

# Determine the false positive and true positive rates
gnb_fpr, gnb_tpr, gnb_threshold = roc_curve(test['class'], gnb.predict(test[feature_names]))

# Determine precision recall
precision_gnb, recall_gnb, _ = precision_recall_curve(test['class'], gnb.predict(test[feature_names]))

#Area Under the precision recall curve
prc_auc_gnb=average_precision_score(test['class'], gnb.predict(test[feature_names]))

#---------------------------------------------------------------------------------------------------------------------------------------------
# Plot of a ROC curve for a specific class
plt.figure()
plt.figure(num=None, figsize=(15, 13), dpi=100, facecolor='w', edgecolor='k')

plt.plot(lreg_fpr, lreg_tpr, label='LREG ROC curve (area = %0.2f)' % auc_lreg)
plt.plot(clf_fpr, clf_tpr, label='CART ROC curve (area = %0.2f)' % auc_clf)
plt.plot(svc_fpr, svc_tpr, label='SVM ROC curve (area = %0.2f)' % auc_svc)
plt.plot(gnb_fpr, gnb_tpr, label='NB ROC curve (area = %0.2f)' % auc_gnb)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------
# Plot Precision-Recall curve
plt.clf()
plt.figure(num=None, figsize=(15, 13), dpi=100, facecolor='w', edgecolor='k')

plt.plot(precision_lreg, recall_lreg, label='LREG Precision-Recall curve (area = %0.2f)' % prc_auc_lreg)
plt.plot(precision_clf, recall_clf, label='CART Precision-Recall curve(area = %0.2f)' % prc_auc_clf)
plt.plot(precision_svc, recall_svc, label='SVM Precision-Recall curve(area = %0.2f)' % prc_auc_svc)
plt.plot(precision_gnb, recall_gnb, label='NB Precision-Recall curve(area = %0.2f)' % prc_auc_gnb)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}')
plt.legend(loc="lower left")
plt.show()
