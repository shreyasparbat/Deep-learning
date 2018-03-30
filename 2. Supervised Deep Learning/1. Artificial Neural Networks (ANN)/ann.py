#%%
## Data Preprocessing

# Importing
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data: Country
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_country = LabelEncoder()
X[:, 1] =  label_encoder_X_country.fit_transform(X[:, 1])

# Encoding categorical data: Gender
label_encoder_X_gender = LabelEncoder()
X[:, 2] =  label_encoder_X_gender.fit_transform(X[:, 1])

# Creating dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling: Independent Variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
## Bulding the ANN

#Importing
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the final layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the traning set (i.e. training it)
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

#%%
## Making Predictions and Evaluating our model

# Predicting Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Accuracy Report (Confusion Matrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Print accuracy
print (float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')

#%%