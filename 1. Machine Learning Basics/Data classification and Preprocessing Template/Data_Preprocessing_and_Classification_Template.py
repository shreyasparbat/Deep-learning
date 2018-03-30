# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:06:42 2017

@author: Shreyas Parbat

Data Preprocessing + Classification Template
"""

#%%
## Importing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
## Data Preprocessing

# Importing dataset
dataset = pd.read_csv("""'path'""")
X = dataset.iloc[:, """cols"""].values
Y = dataset.iloc[:, """cols"""].values

## (IF NEEDED) Taking care of missing of data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = """'NaN'""", strategy = """'mean'""", axis = 0)
#imputer.fit(X[:, """cols"""])
#X[:, """cols"""] = imputer.transform(X[:, """cols"""])

## (IF NEEDED) Encoding categorical data: Independent Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label_encoder_X = LabelEncoder()
#X[:, """cols"""] =  label_encoder_X.fit_transform(X[:, """cols"""])
#onehotencoder = OneHotEncoder(categorical_features = ["""cols from X"""])
#X = onehotencoder.fit_transform(X).toarray()

## (IF NEEDED) Encoding categorical data: Dependent Variable
#label_encoder_Y = LabelEncoder()
#Y = label_encoder_Y.fit_transform(Y)

# Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = """0.2""", random_state = """0""")

# Feature Scaling: Independent Variable
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
trainX = scX.fit_transform(trainX)
testX = scX.transform(testX)

## (IF NEEDED) Feature Scaling: Dependent Variable
#scY = StandardScaler()
#trainY = scY.fit_transform(trainY)
#testY = scY.transform(testY)

#%%
## Building Logical Regression model

# Making classifier and fitting it to training set
#from sklearn.linear_model import LogisticRegression OR some other
#classifier = LogisticRegression(random_state = 0) OR some other
classifier.fit(trainX, trainY)

# Predicting Test set results
predY = classifier.predict(testX)

#%%
## Accuracy report (Confusion Matrix)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY, predY)

# Print accuracy
print (float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')

#%%
## Visualising results

from matplotlib.colors import ListedColormap

#Assumes each pixel point (with resolution 0.01) to be a user.
#Depending on the classifier, colors each pixel red (if prediction = 0) or green (prediction = 1).
#Then, creates red and green dots on the map using the given sets to provide the ground truth.
#Position of each of these points comes from setX, color comes from setY.

setX, setY = trainX, trainY

X1, X2 = np.meshgrid(np.arange(start = setX[:, 0].min() - 1, stop = setX[:, 0].max() + 1, step = 0.01), np.arange(start = setX[:, 1].min() - 1, stop = setX[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(setY)):

    plt.scatter(setX[setY == j, 0], setX[setY == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Linear Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#%%