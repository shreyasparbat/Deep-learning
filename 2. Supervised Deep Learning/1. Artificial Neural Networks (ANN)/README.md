# Code Explanation

Business problem description: predicting a binary out come on the dependent variable (1 = customer leaves bank, 0 = stays).

**Independent variables**: -

1. RowNumber
2. CustomerID
3. Surname
4. CredScore
5. Geography
6. Gender
7. Age
8. Tenure
9. Balance
10. NoOfProducts
11. HasCrCard
12. isActiveMember
13. EstimatedSalary

**Dependent variables**: -

1. Exited

Note: the data set being used here doesn't need to be cleaned as it doesn't have any missing data

## Data Preprocessing

For more information on encoding (and data preprocessing in general), see [data_preprocessing_and_classification_template.py](https://github.com/shreyasparbat/Deep-learning/blob/master/2.%20Supervised%20Deep%20Learning/1.%20Artificial%20Neural%20Networks%20(ANN)/data_preprocessing_and_classification.py).

```python
#%%
## Data Preprocessing

# Importing
import numpy as np
import pandas as pd

# Importing dataset
```

Cols that shouldn't affect churn (and we should exclude): *RowNumber, CustomerID, Surname* -> these are data points that are **too** specific to a certain customer, thus won't be useful when developing a general model.

Cols that definitely affect churn rate (and should be stored in X): *CredScore, Geography, Gender, Age, Tenure, Balance, NoOfProducts, HasCrCard, isActiveMember, EstimatedSalary*.

Col which is dependent on the other cols (dependent variable, stored in Y): *Exited*

Note - 3:13 because upper bound excluded

```python
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data: Country
```

Note: Even though the Dependent variable is categorical, we do not need to encode it cause it is in numerical values of 0 & 1

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_country = LabelEncoder()
X[:, 1] =  label_encoder_X_country.fit_transform(X[:, 1])

# Encoding categorical data: Gender
label_encoder_X_gender = LabelEncoder()
X[:, 2] =  label_encoder_X_gender.fit_transform(X[:, 1])

# Creating dummy variables
```

This needs to be done only for country as it contains 3 categories. Gender contains 2 categories, so no need to create dummy variables 

```python
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
```

Here, remove the first column (i.e. one of the countries) to avoid falling into the [dummy variable trap](https://www.moresteam.com/WhitePapers/download/dummy-variables.pdf) (basically, if both France and Spain are 0, then country **has** to be Germany).

```python
X = X[:, 1:]

# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling: Independent Variables
```

Is feature Scaling really required in Deep learning? **Yes, compulsorily** because a lot of computation power goes into deep learning and feature scaling eases these calculations.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Building the ANN

Steps to training an ANN with Stochastic Gradient Descent (more info in notes): -

1. Randomly initialise the weights to small numbers close to 0 (but not 0)
2. Input the first observation of your dataset in the input layer, each feature in one input node
3. Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until you get the predicted result y
4. Compare the predicted result to the actual result. Measure the generated error
5. Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights
6. Two options: -
   - Reinforcement Learning: Repeat steps 1-5 and update the weights after each observation
   - Batch Learning: Repeat steps 1-5 but update the weights only after a batch of observations
7. When the whole training set has passed through the ANN, an Epoch is over. Redo for epochs

```python
#%%
## Bulding the ANN

#Importing
import keras
```

**Sequential**: Required to initialise our ANN

```python
from keras.models import Sequential
```

**Dense**: Required to build the layers of our ANN

```python
from keras.layers import Dense

# Initialising the ANN
```

Two ways of doing this: -

1. Defining the sequence of layers
2. Defining a graph

Here, we're making an ANN with successive layers, so we'll initialise our ANN by defining a sequence of layers (option 1).

Note: 'classifier' *is* the neural network!

Another note: we don't put in any arguments in 'Sequential()' cause we will define the input, first hidden layer, remaining hidden layers and output layer manually.

```python
classifier = Sequential()

# Adding input layer and first hidden layer
```

Number of nodes in the input layer = number of independent variables

For hidden layers, we will use **Rectifier Function**: -

![](images\1522406955451.png)

**add()**

1. Parameter: the layers to be added to the ANN

**Dense()**

1. output_dim: # of nodes to be added to the hidden layer (by adding this, your also specifying the number inputs). Two ways of choosing this #: -

   - Follow this tip: # = average of the number of nodes in input layer and output layer
   - Parameter tuning: using techniques like K4 cross validation (See CNN).

   Parameter tuning is the way to go (see CNN), but here we will just follow the tip.

2. init: Specifies how the parameters will be initialised. 'uniform' initialises them according to the uniform distribution with weights close to 0.

3. activation: which activation function to use. *'relu' = Rectifier Function*, *'sigmoid' = Sigmoid Function*

4. input_dim: specifies # of nodes in the input layer. Here, its the # of independent variables as this is just the first layer. Also it is compulsory to add it here because so far the ANN has only been initialised, so this layer doesn't know which layer it should be expecting as input.

   Not adding this will give the following error: "The first layer in a Sequential model must get an 'input_shape' or 'batch_input_shape' argument".

   We won't need to specify this argument from the next hidden layer.

   Set # = number of input variables!

```python
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding second hidden layer
```

Not really needed for our dataset, but we'll add more here anyway for practice.

Same logic for choosing 'output_dim' as before.

```python
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the final layer
```

For output layer, we will be using **Sigmoid Function** (great for outputting probabilities): -

![](images\1522407035343.png)

Here, *output_dim = 1* (since we want a binary outcome)

Note: if dependent variable has 3 or more categories (unlike here), use the *softmax function*.

```python
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
```

Basically means applying stochastic gradient to our ANN

**compile()**

1. optimizer: [Stochastic gradient optimising Algo](http://ruder.io/optimizing-gradient-descent/) to be used to define the optimal set of weights in NN (till now they have only been uniformly initialised).

   Here, we choose the  *adam* algo

2. loss: the loss function to be used within the chosen algo (that will be optimised in order to optimise the weights). Examples: -

   - Ordinary least squares

     - $$
       min(SUM(y-y^2))
       $$

     - When to use: Simple Linear Regression

   - Logarithmic Loss

     - ![](images\1522413445690.png)
     - When to use: when activation function is the **sigmoid function**

   Here, since our activation function for our output layer is the sigmoid function, here we will use **log loss function**.

   Note: if dependent variable has binary outcome, use *binary_crossentropy*. Else, use *categorical_crossentropy*.

3. metrics: specify which criteria to use to evaluate the module (in list form). Typically, we use *accuracy*

```python
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the traning set (i.e. training it)
```

**fit()**

1. X_train
2. Y_train
3. batch_size: # of observations after which the weights will be updated. No rule for choosing this, must experiment
4. nb_epoch: # of epochs that will take place. Again, no rule for choosing this, must experiment

```python
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)
```

## Making Predictions and Evaluating

```python
#%%
## Making Predictions and Evaluating our model

# Predicting Test set results
```

*keras* has the same *predict()* method as *sklearn*.

Note: in the last line we change the prediction to *true* (if probability of leaving = y > threshold = 0.5) else *false*. We can change this threshold to whatever we see fit

```python
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Accuracy Report (Confusion Matrix)
```

Comparing predicted churn probabilities of all customers in the test set to the *actual* churn result of those customers

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Print accuracy
print (float(cm[0][0] + cm[1][1])/np.sum(cm) * 100, '%')

#%%
```

Outcome: **accuracy = 86.25%**. Pretty cool! This could potentially be improved using parameter tuning.

The bank can now run all of its clients through this ANN and get a list, with each row being the probability of that customer leaving the bank (or they could do the same as us and get absolute values using a certain threshold). This has a lot of potential business value!
