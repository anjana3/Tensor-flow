# Project Title
### Predict house prices: regression

## Objective
In a regression problem, we aim to predict the output of a continuous value, like a price or a probability. 

## Prerequisites
Tensor-flow,python.

## Installing
 
1. Visual studio code (to run the code)
for this you need to set up your own enviroment to load the packages
1. packages are required to run the model
import numpy as np

import pandas as pd

import pydot

from sklearn import tree

import matplotlib.pyplot as plt

from tensorflow import keras

# Data set to download

> https://1.salford-systems.com/Portals/160602/Tutorial%20Datasets/boston.xls

## Details of data set

Independent Variables(features) :

The data set contains 13 different features:

1. Per capita crime rate.
1. The proportion of residential land zoned for lots over 25,000 square feet.
1. The proportion of non-retail business acres per town.
1. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
1. Nitric oxides concentration (parts per 10 million).
1. The average number of rooms per dwelling.
1. The proportion of owner-occupied units built before 1940.
1. Weighted distances to five Boston employment centers.
1. Index of accessibility to radial highways.
1. Full-value property-tax rate per $10,000.
1. Pupil-teacher ratio by town.
1. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
1. Percentage lower status of the population.

## Dependent Variable(Target Variable ) :

1. The labels are the house prices in thousands of dollars. (You may notice the mid-1970s prices.)




# Train the model using Sequential model :

It build with two densely connected hidden layers, and an output layer that returns a single, continuous value.

## Normalize features

It's recommended to normalize features that use different scales and ranges. For each feature, subtract the mean of the feature and divide by the standard deviation.

## Predict

Finally, predict some housing prices using data in the testing set:

###Inference
.Testing set Mean Abs Error: $2887.38
[24.272364  27.563288  23.476702  10.903258  20.71052   21.05875
 22.914602  21.769543  23.047016  22.145649   6.298267  13.038064
 16.735735   7.6383185 42.764328  39.674206  23.69914   41.500324
 33.707645  22.578915  25.533892  22.818241  20.520887  27.556765

## Conclusion
1. Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
1. Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
1. When input data features have values with different ranges, each feature should be scaled independently.
1. If there is not much training data, prefer a small network with few hidden layers to avoid over fitting.
Early stopping is a useful technique to prevent overfitting.s

## Author

**K. ANJANA NARAYANA**

## Acknowledgements

`https://www.tensorflow.org/tutorials/keras/basic_regression#the_boston_housing_prices_dataset`

`https://www.youtube.com/watch?v=yX8KuPZCAMo`




