# DATA 630 Assignment 4
# Written by Daanish Ahmed
# Semester Summer 2017
# July 14, 2017
# Professor Edward Herranz

# This R script involves the creation of three neural networks using a dataset 
# on heart disease information.  The purpose of this assignment is to build a 
# model that accurately predicts the class of the dependent variable "FSTAT," 
# which represents a patient's vital status at the time of their last follow-
# up session.  I will experiment with using different numbers of hidden layers
# and hidden layer nodes in each model.  At the end, I will evaluate the 
# accuracies of the three models to determine which set of parameters will 
# produce the most accurate predictions.



# This section of code covers opening the dataset and initializing the packages 
# that are used in this script.

# Sets the working directory for this assignment.  Please change this directory 
# to whichever directory you are using, and make sure that all files are placed 
# in that location.
setwd("~/Class Documents/2016-17 Summer/DATA 630/R/Assignment 4")

# In order to run the neural network commands, we need to install the neuralnet 
# package:

# If you have not installed this package yet, remove the # symbol below.
# install.packages("neuralnet")

# Loads the neuralnet package into the system.
library("neuralnet")

# Opens the CSV file "whas1.csv".
heart_dis <- read.csv("whas1.csv", head=TRUE, sep=",")

# End of opening the dataset.



# This section of code covers data preprocessing.  It includes exploration of 
# the original dataset, removing variables, and dealing with missing values.

# Previews the heart disease dataset.
View(heart_dis)

# Displays the structure of the data.  This is necessary to see if there are 
# any unique identifiers (IDs) that can be removed.  Such variables are not 
# useful for the analysis and should be removed.
str(heart_dis)

# The first variable is an ID, and we remove it.
heart_dis <- heart_dis[, -1]

# Verifies that the ID variable has been removed.
str(heart_dis)

# Displays the descriptive statistics for all variables in the dataset.  This 
# shows whether all variables are numeric and if there are any missing values.
summary(heart_dis)

# Since all variables are numeric and there are no missing values, we do not 
# need to worry about these issues.

# Sets the input variables to the same scale, such that they have the same
# mean and standard deviation.
heart_dis[1:12] <- scale(heart_dis[1:12])

# Verifies that the variables are set to the same scale.  All variables 
# except for "fstat" should have a mean equal to 0.
summary(heart_dis)

# End of data preprocessing.



# This section of code covers the creation of the first neural network model. 
# It includes dividing the data into training and test datasets, creating and 
# visualizing the model, and analyzing the model on the training and test 
# datasets.

# Generates a random seed to allow us to reproduce the results.
set.seed(12345)

# The following code splits the dataset into a training set consisting of 
# 70% of the data and a test set containing 30% of the data.
ind <- sample(2, nrow(heart_dis), replace = TRUE, prob = c(0.7, 0.3))
train.data <- heart_dis[ind == 1, ]
test.data <- heart_dis[ind == 2, ]

# This command implements the neural network method on training data using 
# class as the dependent variable and all other variables as independent 
# variables.  We use one hidden layer with 2 nodes, and linear output set to 
# false for classification models.
nn <- neuralnet(formula = FSTAT ~ AGE + SEX + CPK + SHO + CHF + MIORD + 
                  MITYPE + YEAR + YRGRP + LENSTAY + DSTAT + LENFOL, 
                data = train.data, hidden = 2, 
                err.fct = "ce", linear.output = FALSE)

# Shows the summary of the model.
summary(nn)

# We will examine the following properties: the response values, result 
# matrix, and net result probabilities.

# Shows the values of the dependent variable for the first 20 observations.
nn$response[1:20]

# Shows the probability for the first 20 records.
nn$net.result[[1]][1:20]

# Shows the network result matrix, which includes information on the number 
# of training steps, the error, and the weights.
nn$result.matrix

# Creates a visualization of the neural network.
plot(nn)

# Computes the predicted values for the training set and rounds them to the 
# nearest integer (either 0 or 1).
mypredict <- compute(nn, nn$covariate)$net.result
mypredict <- apply(mypredict, c(1), round)

# Shows the first 20 predicted values.
mypredict[1:20]

# Creates the confusion matrix on the training data.
table(mypredict, train.data$FSTAT, dnn =c("Predicted", "Actual"))

# Computes the predicted values for the test set and rounds them to the 
# nearest integer (either 0 or 1).
testPred <- compute(nn, test.data[, 0:12])$net.result
testPred <- apply(testPred, c(1), round)

# Creates the confusion matrix on the test data.
table(testPred, test.data$FSTAT, dnn = c("Predicted", "Actual"))

# End of creating the first neural network model.



# This section of code covers the creation of the second neural network 
# model.  This model will have two hidden layers, one with 4 nodes and one 
# with 2 nodes.  Everything else will stay the same.  It will require many 
# of the same steps as the previous model.

# Generates a random seed to allow us to reproduce the results.
set.seed(12345)

# Implements the neural network model on the same training set using the 
# same variables as input and output parameters.  The only difference is 
# that there are now two hidden layers, one with 4 nodes and one with 2 
# nodes.
nn <- neuralnet(formula = FSTAT ~ AGE + SEX + CPK + SHO + CHF + MIORD + 
                  MITYPE + YEAR + YRGRP + LENSTAY + DSTAT + LENFOL, 
                data = train.data, hidden = c(4, 2), 
                err.fct = "ce", linear.output = FALSE)

# Shows the probability for the first 20 records.
nn$net.result[[1]][1:20]

# Shows the network result matrix, which includes information on the number 
# of training steps, the error, and the weights.
nn$result.matrix

# Creates a visualization of the neural network.
plot(nn)

# Computes the predicted values for the training set and rounds them to the 
# nearest integer (either 0 or 1).
mypredict <- compute(nn, nn$covariate)$net.result
mypredict <- apply(mypredict, c(1), round)

# Creates the confusion matrix on the training data.
table(mypredict, train.data$FSTAT, dnn =c("Predicted", "Actual"))

# Computes the predicted values for the test set and rounds them to the 
# nearest integer (either 0 or 1).
testPred <- compute(nn, test.data[, 0:12])$net.result
testPred <- apply(testPred, c(1), round)

# Creates the confusion matrix on the test data.
table(testPred, test.data$FSTAT, dnn = c("Predicted", "Actual"))

# End of creating the second neural network model.



# This section of code covers the creation of the third neural network 
# model.  This model will only have one hidden layer with a single node. 
# Everything else will stay the same.  It will require many of the same 
# steps as the previous model.

# Generates a random seed to allow us to reproduce the results.
set.seed(12345)

# Implements the neural network model on the training set using the same 
# variables as input and output parameters.  There is now only one hidden 
# layer with one node.
nn <- neuralnet(formula = FSTAT ~ AGE + SEX + CPK + SHO + CHF + MIORD + 
                  MITYPE + YEAR + YRGRP + LENSTAY + DSTAT + LENFOL, 
                data = train.data, hidden = 1, 
                err.fct = "ce", linear.output = FALSE)

# Shows the probability for the first 20 records.
nn$net.result[[1]][1:20]

# Shows the network result matrix, which includes information on the number 
# of training steps, the error, and the weights.
nn$result.matrix

# Creates a visualization of the neural network.
plot(nn)

# Computes the predicted values for the training set and rounds them to the 
# nearest integer (either 0 or 1).
mypredict <- compute(nn, nn$covariate)$net.result
mypredict <- apply(mypredict, c(1), round)

# Creates the confusion matrix on the training data.
table(mypredict, train.data$FSTAT, dnn =c("Predicted", "Actual"))

# Computes the predicted values for the test set and rounds them to the 
# nearest integer (either 0 or 1).
testPred <- compute(nn, test.data[, 0:12])$net.result
testPred <- apply(testPred, c(1), round)

# Creates the confusion matrix on the test data.
table(testPred, test.data$FSTAT, dnn = c("Predicted", "Actual"))

# End of creating the third neural network model.

# End of script.

