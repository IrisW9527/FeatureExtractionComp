import numpy as np
from sklearn.naive_bayes import GaussianNB # classifier 1
import sklearn.discriminant_analysis
import pandas as pd
from sklearn.model_selection import train_test_split # cross-validation
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC # classifier 2
from time import time
from sklearn.metrics import accuracy_score

# import data sets - red wine - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1598 x 12 except for the headers

# HEADERS:
# INPUT VARIABLES: - - - - - - - - - - - - - - - - - - -
# 0 "fixed acidity"; 1 "volatile acidity"; 2 "citric acid"; 3 "residual sugar"; 4 "chlorides"; 5 "free sulfur dioxide";
# 6 "total sulfur dioxide"; 7 "density"; 8 "pH"; 9 "sulphates"; 10 "alcohol";
# OUTPUT VARIABLE: - - - - - - - - - - - - - - - - - - -
# 11 "quality"

X=pd.read_csv("winequality-red.csv",header = 0) #THANK GOD THE FREAKING HEADER
# print(X.isnull().sum()) # checking any null value existing
# print(X['quality'].unique()) # checking the "quality" values
# print(X["quality"].value_counts()) # count the "quality" values

X = X.to_numpy() # convert the data to a matrix
print('X = ', X)

# separate inputs and output - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# mean = X.mean(axis = 0) # size 1 x 12
X_input = X[:, :-1] # column 0 to column 10 - 11 variable columns
Y_output = X[:,-1] # the last column - column 11
print('X_inputs = ', X_input)
print('Y_outputs = ', Y_output)

# classify output into a binary class
for k in range(1598):
    if Y_output[k] < 6.5:
        Y_output[k] = 0
    else:
        Y_output[k] = 1
print('Y_output_binary = ', Y_output)

# Training the data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# X: data - inputs
# Y: target - output

X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_output, test_size = 0.25)
# Y_train = np.array(Y_train)
# Y_test = np.array(Y_test)

print('X_train = ', X_train) # 3/4 of the inputs data are trained
print('X_test = ', X_test) # 1/4 of the inputs data
print('Y_train = ', Y_train) # 3/4 of the output data
print('Y_test = ', Y_test) # 1/4 of the output data
print('Y_train_shape: ', Y_train.shape)


# processing data: mean center and scale - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# for input training data - - - - - - - - - - - -
mean_X_train = np.mean(X_train) # size 1 x 11
print('mean_X_train = ', mean_X_train)
std_X_train = np.std(X_train) # size 1 x 11
print('std_X_train = ', std_X_train)
X_train_centered = (X_train - mean_X_train)/std_X_train # normalized dataset
print('X_train_centered = ', X_train_centered)

# for input testing data  - - - - - - - - - - - -
# use the mean and std of training data
X_test_centered = (X_test - mean_X_train)/std_X_train # normalized dataset
print('X_train_centered = ', X_test_centered)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - -
# PCA FEATURES:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# : PCA for the input TRAINING data - - - - - - - - - - - - - - - - - - - - - -
# compute the eigenvalues and eigenvectors of the covariance matrix
U_train, V_train = np.linalg.eig(np.dot(X_train_centered.T,X_train_centered))
print('U_train = ', U_train) # size 1 x 11
print('V_train = ', V_train) # size 11 x 11

# reduce the dimensionality with PCA
# 11 input variables in total, we will experiment reducing dimensionality to ~? variables

for iteration1 in range(8):  # to which degree the dimension is reduced to ~ = 4 (CHANGE NUMBER HERE TO EXPERIMENT)
    minEigValIndex1 = np.argmin(U_train)
    newV_train = np.zeros((11,11-iteration1)) # size of new eigenvectors of zeros: 11 x 11

    index = 0
    for i in range(11-iteration1):
        if (i != minEigValIndex1):
            newV_train[:,index] = V_train[:,i] # size of newV:
            index = index + 1

    U_train = np.delete(U_train,minEigValIndex1)

print('U_train_new = ', U_train) # size 1 x (11-iteration)
print('newV_train = ', newV_train) #???? FIGURE OUT THE SIZE !!! probably because of one var is not singular

newXScores_train_pca = np.dot(X_train, newV_train)
print('newXScores_train = ', newXScores_train_pca)

# : PCA for the input TESTING data - - - - - - - - - - - - - - - - - - - - - -
# compute the eigenvalues and eigenvectors of the covariance matrix
U_test, V_test = np.linalg.eig(np.dot(X_test_centered.T,X_test_centered))
print('U_test = ', U_test) # size 1 x 11
print('V_test = ', V_test) # size 11 x 11

# reduce the dimensionality with PCA
# 11 input variables in total, we will experiment reducing dimensionality to ~? variables

for iteration2 in range(8):  # to which degree the dimension is reduced to ~ = 4 (CHANGE NUMBER HERE TO EXPERIMENT)
    minEigValIndex2 = np.argmin(U_test)
    newV_test = np.zeros((11,11-iteration2)) # size of new eigenvectors of zeros: 11 x 11

    index = 0
    for j in range(11-iteration2):
        if (j != minEigValIndex2):
            newV_test[:,index] = V_test[:,j] # size of newV:
            index = index + 1

    U_test = np.delete(U_test,minEigValIndex2)

print('U_test_new = ', U_test)
print('newV_test = ', newV_test)

newXScores_test_pca = np.dot(X_test, newV_test)
print('newXScores_train = ', newXScores_test_pca)

# APPLY CLASSIFIERS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

print('Y_test = ', Y_test)
print('Y_test_shape: ', Y_test.shape)

# classifier 1: Support Vector Machine - - - - - - - - - - - -
svm_pca = LinearSVC()
t_SVM_pca_TrainingTime=time()
svm_pca.fit(newXScores_train_pca,Y_train)
print("training time of SVM:", round(time()-t_SVM_pca_TrainingTime, 7), "s") # 5 is the floating number

t_SVM_pca_TestingTime=time()
Y_predicted_SVM_pca = svm_pca.predict(newXScores_test_pca)
print("testing time of SVM:", round(time()-t_SVM_pca_TestingTime, 7), "s") # 5 is the floating number

print('Y_predicted = ', Y_predicted_SVM_pca)
print('Y_predicted_shape: ', Y_predicted_SVM_pca.shape)
accuracy_SVM_pca = accuracy_score(Y_test, Y_predicted_SVM_pca)
print('Accuracy of the SVM Classifier is ', accuracy_SVM_pca)

# confusion matrix
# predicted_SVM_pca = np.array(Y_predicted_SVM_pca)
tn_SVM_pca, fp_SVM_pca, fn_SVM_pca, tp_SVM_pca = confusion_matrix(Y_test, Y_predicted_SVM_pca).ravel()
print(tn_SVM_pca, fp_SVM_pca, fn_SVM_pca, tp_SVM_pca)# classifier computation time

# classifier 2: Naive Bayes - - - - - - - - - - - -
gnb_pca = GaussianNB() # create a NB classifier
t_NB_pca_TrainingTime = time() # computation time
gnb_pca.fit(newXScores_train_pca,Y_train) # train the model using the training datasets
print("training time of NB:", round(time()-t_NB_pca_TrainingTime, 7), "s")

t_NB_pca_TestingTime = time() # computation time
Y_predicted_NB_pca = gnb_pca.predict(newXScores_test_pca)
print("testing time of NB:", round(time()-t_NB_pca_TestingTime, 7), "s")

print('Y_predicted = ', Y_predicted_NB_pca)
print('Y_predicted_shape: ', Y_predicted_NB_pca.shape)
accuracy_NB_pca = accuracy_score(Y_test, Y_predicted_NB_pca)
print('Accuracy of the Naive Bayes Classifier is ', accuracy_NB_pca)

# confusion matrix
# predicted_NB_pca = np.array(Y_predicted_NB)
tn_NB_pca, fp_NB_pca, fn_NB_pca, tp_NB_pca = confusion_matrix(Y_test, Y_predicted_NB_pca).ravel()
print(tn_NB_pca, fp_NB_pca, fn_NB_pca, tp_NB_pca)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# RAW FEATURES VIA BACKWARD SEARCH
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# for input training data - - - - - - - - - - - - - - -
bestFeature_train = 100 * np.ones(8) # create an empty array for store index
# print('bestFeature_train = ',bestFeature_train)

X_test_selection = np.zeros((400, 11))  # create an empty matrix for

for selection in range(8):

    # if selection == 0:
    #     X_train_selection = np.zeros((1198, 1))
    # else:
    #     X_train_selection = np.concatenate((X_train_selection, np.zeros((1198, 1))),axis = 1)

    X_train_selection = np.zeros((1198, 11))

    errors_train = 100 * np.zeros(11)

    for feature in range(11):
        if(not(feature == bestFeature_train[0]) or (feature == bestFeature_train[1]) or (feature == bestFeature_train[2]) or (feature == bestFeature_train[3]) or (feature == bestFeature_train[4]) or (feature == bestFeature_train[5]) or (feature == bestFeature_train[6]) or (feature == bestFeature_train[7])):
            X_train_selection[:,selection] = X_train_centered[:,feature]

            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()  # create a LDA object
            lda.fit(X_train_selection, Y_train)  # learning the projection matrix
            prediction_train = lda.predict(X_train_selection)  # gives the predicted labels for each variable in X

            errors_train[feature] = sum(abs(Y_train - prediction_train)) # calculate the error
            print("total error with all features (backward search) = ", errors_train[feature])

    bestFeature_train[selection] = np.argmin(errors_train) # the index of the smallest error term
    X_train_selection[:, selection] = X_train_centered[:, int(bestFeature_train[selection])]
    X_test_selection[:,selection] = X_test_centered[:, int(bestFeature_train[selection])]
    print('X_train_selection = ', X_train_selection)
    print('X_test_selection = ', X_test_selection)

print('bestFeature_train = ', bestFeature_train)

# print('X_train_centered = ', X_train_centered)
print('done')

# APPLY CLASSIFIERS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
X_train_selection_featured = X_train_selection[:,0:8]
print('X_test_selection = ', X_train_selection_featured)
X_test_selection_featured = X_test_selection[:,0:8]
print('X_test_selection = ', X_test_selection_featured)

# classifier 1: Support Vector Machine - - - - - - - - - - - -
svm_featured = LinearSVC()
t_SVM_featured_TrainingTime = time()
svm_featured.fit(X_train_selection_featured,Y_train) # training time
print("training time of SVM:", round(time()-t_SVM_featured_TrainingTime, 7), "s") # 5 is the floating number

t_SVM_featured_TestingTime = time()
Y_predicted_SVM_featured = svm_featured.predict(X_test_selection_featured) # testing time
print("testing time of SVM:", round(time()-t_SVM_featured_TestingTime, 7), "s") # 5 is the floating number

print('Y_predicted_SVM_featured = ', Y_predicted_SVM_featured)
print('Y_test = ', Y_test)
print('Y_predicted_SVM_featured_shape: ', Y_predicted_SVM_featured.shape)
accuracy_SVM_featured = accuracy_score(Y_test, Y_predicted_SVM_featured)
print('Accuracy of the SVM Classifier is ', accuracy_SVM_featured)

# confusion matrix
print('confusion matrix of SVM in feature selection: ')
tn_SVM, fp_SVM, fn_SVM, tp_SVM = confusion_matrix(Y_test, Y_predicted_SVM_featured).ravel()
print(tn_SVM, fp_SVM, fn_SVM, tp_SVM)# classifier computation time


# classifier 2: Naive Bayes - - - - - - - - - - - -
gnb_featured = GaussianNB() # create a NB classifier
t_NB_featured_TrainingTime = time() # computation time
gnb_featured.fit(X_train_selection_featured,Y_train) # train the model using the training datasets
print("training time of NB:", round(time()-t_NB_featured_TrainingTime, 7), "s")

t_NB_featured_TestingTime = time() # computation time
Y_predicted_NB_featured = gnb_featured.predict(X_test_selection_featured)
print("testing time of NB:", round(time()-t_NB_featured_TestingTime, 7), "s")

print('Y_predicted = ', Y_predicted_NB_featured)
print('Y_test = ', Y_test)
print('Y_predicted_shape: ', Y_predicted_NB_featured.shape)
accuracy_NB_featured = accuracy_score(Y_test, Y_predicted_NB_featured)
print('Accuracy of the Naive Bayes Classifier is ', accuracy_NB_featured)

print('confusion matrix of NB in feature selection: ')
tn_SVM, fp_SVM, fn_SVM, tp_SVM = confusion_matrix(Y_test, Y_predicted_NB_featured).ravel()
print(tn_SVM, fp_SVM, fn_SVM, tp_SVM)# classifier computation time