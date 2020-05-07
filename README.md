# Feature Extraction Performance Comparisons between PCA and feature selection

Introduction

The dataset includes 1598 observations and 11 input variables that are “fixed acidity”,
“volatile acidity”, “citric acid”, “residual sugar”, “chlorides”, “free sulfur dioxide”, “total sulfur
dioxide”, “density”, “pH”, “sulphates”, and “alcohol” as well as 1 output “quality”.

Methodology

The data is firstly pre-processed into training data and testing data in 4 folds. All variables are
numerical, and the output variable is classified into a binary class by simply setting a threshold
value to the output variable “quality”, which has a series of values ranging from 4 to 8. The
mean of the training data is then calculated and applied to the testing data as well because the
testing data shouldn’t be biased by the training data. In this project, Support Vector Machine
and Naïve Bayes classifiers are used in both PCA and raw data feature selection experiments.
The number of the features in both methods is varied in order to compare the results in terms
of data accuracy.
For PCA experiments, the eigenvalues and eigenvectors are derived from the covariance matrix
of the input training data. Then the reduced dimensionality is experimented and the new scores
data is fed into the classifiers. Computational time and confusion matrix are then can be
obtained.
For the raw data feature selection via search method, here backward search is used. The
number of the features is experimented here as well. within the experiment, Linear
Discriminant Analysis is conducted to get the smallest error when filtering the features. After
that the input training data is trained by the classifiers and the predicted data can be found.
