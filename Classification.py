

################################# DATA ANALYSIS #################################

# 1. LOAD DATA AND SHOW ITS SUMMARY
import ax as ax
import pandas as p
import numpy as np
import csv

link = "http://archive.ics.uci.edu/ml/machine-learning-databases/winequality/winequality-white.csv"

def readfile(file):     # fonction pour convertir le fichier en liste
    f= open(file,'r')
    strlist= f.read()
    strlist= strlist.split('\n')

    final = []

    for element in strlist:
        strelement = []
        strchamp = element.split(",")

        for row in strchamp:
            strelement.append(row)

        final.append(strelement)

    return(final)

fichier = "resources/winequality-white.csv"
f = readfile(fichier)

df =p.DataFrame(f)

print("\n========== Dataset Summary ===========\n")
df.info()
print("\n========== A few first samples ===========\n")
print(df.head())

# We have 12 samples

print("Nous sommes ici  \n",df[11])



# 2. What are the wine qualities and the related number of samples?
df = df[:-1] # car la dernière ligne était nulle
X = df.drop(columns=[11]) # we drop the column "quality"
Y = df[11]
print("\n =========== WINE QUALITY =========== \n")
print(Y.value_counts())


# 3. GROUP DATA BY QUALITY LEVEL

print(Y[1:5])
print("AVANT :", type(Y[2]))
print(Y[0])
# on recode en INT :
i = 0
for a in Y :
    if Y[i] == "quality":
        print(Y[i])
    else:
        try:
            Y[i] = int(a)
        except:
            pass
    i +=1

print("APRÈS :" ,type(Y[2]))

#bad wine (y=0) : quality <= 5 and good quality (y=1) otherwise
#Y.loc[np.where(Y<6)] = 0
#Y.loc[np.where(Y>5)] = 1

def transformation(x):
    if x <6 :
        return 0
    elif x >5 :
        return 1

i = 0
for a in Y:
    try:
        Y[i] = transformation(Y[i])
    except :
        pass
    i+=1

Y.name = "Label"

# 4. PERFORM A STATISTICAL ANALYSIS OF THE INPUT VARIABLES

import matplotlib.pyplot as plt
import seaborn as sns

# je me rends compte que les données sont au format str
# donc il faut les recoder en float
print("Type :\n", type(X[5][2]))

X[1:] = X[1:].astype(float)



plt.figure()
sns.boxplot(data=X,orient="v",palette="Set1",width=1.5, notch=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.figure()
corr = X.corr()
sns.heatmap(corr)
