#question 1
import pandas as pd
import numpy as np
link = "resources/winequality-white.csv"
df = pd.read_csv(link, header="infer", delimiter=",")
print("\n========= Dataset summary ========= \n")
df.info()
print("\n========= A few first samples ========= \n")
print(df.head())
print('number of samples', df.shape[0])
print('number of variables', df.shape[1])



#question 2
X = df.drop("quality", axis=1) #we drop the column "quality"
Y = df["quality"]
print("\n========= Wine Qualities ========= \n")
print(Y.value_counts())

#question 3
# bad wine (y=0) : quality <= 5 and good quality (y= 1) otherwise
Y.loc[np.where(Y <= 5)]=0
Y.loc[np.where(Y > 5)]=1
Y.name = "Label"


#question 4
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.boxplot(data=X,orient="v",palette="Set1",width=1.5, notch=True)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.figure()
corr = X.corr()
sns.heatmap(corr)


#2.1
from sklearn.model_selection import train_test_split
Xa, Xt, Ya, Yt = train_test_split(X, Y, shuffle=True, test_size=1/3, stratify=Y)
Xt, Xv, Yt, Yv = train_test_split(Xt, Yt, shuffle=True, test_size=0.5, stratify=Yt)
print('learning set', Xa.shape[0])
print('test set', Xt.shape[0])
print('validation set', Xv.shape[0])



#2.2.1
from sklearn.neighbors import KNeighborsClassifier
# Fit the model on (Xa, Ya)
k = 2
clf = KNeighborsClassifier(n_neighbors = k)
clf.fit(Xa, Ya)
# Predict the labels of samples in Xv
Ypred_v = clf.predict(Xv)
# evaluate classification error rate
from sklearn.metrics import accuracy_score
error_v = 1-accuracy_score(Yv, Ypred_v)
print('error', error_v)


#2.2.2
k_values = np.arange(1,40)
clfs = [KNeighborsClassifier(n_neighbors = k) for k in k_values]
for clf in clfs:
    clf.fit(Xa, Ya)
train_error_values = np.array([ 1 - accuracy_score(Ya, clf.predict(Xa)) for clf in clfs])
test_error_values = np.array([ 1 - accuracy_score(Yt, clf.predict(Xt)) for clf in clfs])
validation_error_values = np.array([ 1 - accuracy_score(Yv, clf.predict(Xv)) for clf in clfs])

print('learning curves')
plt.plot(train_error_values, 'b')
plt.plot(test_error_values, 'g')
plt.plot(validation_error_values, 'r')
# when k is small overfitting
# when k is big underfitting
minIndex = np.argmin(test_error_values)
print('k* is', k_values[minIndex])
print('test error for k* is', test_error_values[minIndex])
print('validation error for k* is', validation_error_values[minIndex])


#2.3.1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=True, with_std=True)
sc = sc.fit(Xa)
Xa_n = sc.transform(Xa)
Xv_n = sc.transform(Xv)
Xt_n = sc.transform(Xt)


#2.3.2
k_values = np.arange(1,40)
clfs = [KNeighborsClassifier(n_neighbors = k) for k in k_values]
for clf in clfs:
    clf.fit(Xa_n, Ya)
train_error_values = np.array([ 1 - accuracy_score(Ya, clf.predict(Xa_n)) for clf in clfs])
test_error_values = np.array([ 1 - accuracy_score(Yt, clf.predict(Xt_n)) for clf in clfs])
validation_error_values = np.array([ 1 - accuracy_score(Yv, clf.predict(Xv_n)) for clf in clfs])

print('learning curves')
plt.plot(train_error_values, 'b')
plt.plot(test_error_values, 'g')
plt.plot(validation_error_values, 'r')

minIndex = np.argmin(test_error_values)
print('k* is', k_values[minIndex])
print('test error for k* is', test_error_values[minIndex])
print('validation error for k* is', validation_error_values[minIndex])
#ramark error rate will be much lower



"""
for question 3 see this
https://dzone.com/articles/machine-learning-validation-techniques
the idea is to use other validation techniques rather than splitting into train and test
see espically K Fold cross validation
"""


