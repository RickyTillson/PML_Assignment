---
title: "Practical Machine Learning - Weight Lifting Excercise"
author: "Ricky Tillson"
date: "December 2014"
output: pdf_document
---

## Introduction

Wearable tedhchnology such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* make it possible to collect large amounts of data regarding personal activity. Most of this is aimed towards recording activity levels, improving health or finding patterns in their behavior. This data usually quantifies *how much* of an activity is carried out, but not *how well* it is performed.

The aim of this project is to build a machine learning model from a training set of data and use this to predict a testing set of data.


## Data

The training data is available at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data is available at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Full details about the data set is available at http://groupware.les.inf.puc-rio.br/har

The data is taken from a series of 6 individuals carrying out unilateral dumbell bicep curls in one of 5 methods:

- A - correctly

- B - throwing the elbows to the front

- C - only lifting the dumbell halfway

- D - only returning the dumbell half way

- E - throwing the hips forwards

Measurements were taken using Razor interial measurement units which measure, in three axes, acceleration, gyroscope and magnetometer data. Each of the 6 participants carried out 10 repetitions of the excercise according to the specification. There are some measurements (min, max, standard deviation etc.) taken which summarise the repetitions.

``` {r get_data, cache = TRUE}

train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(train_url, destfile = "training_file.csv", method = "curl")
download.file(test_url, destfile = "testing_file.csv", method = "curl")
train_data <- read.csv(file = "training_file.csv", header = TRUE)
test_data <- read.csv(file = "testing_file.csv", header = TRUE)
dim(train_data)
```

The data needs partitioning so that we have both a training and a validation set (a seed has been used so that this is fully reproducible).

``` {r partition, cache = TRUE, warning = FALSE}
library(caret)
set.seed(12345)
trainset <- createDataPartition(train_data$classe, p = 0.8, list = FALSE)
validation_data <- train_data[-trainset,]
train_data <- train_data[trainset,]
```

We need to try and reduce the data set before trying to build a model. The first thing to do is remove the first columns which contain only data relating to time, date, participant and row number, and any columns relating to aggregating data.

``` {r cleaning_one, cache = TRUE}
train_data_cleaned <- train_data[,-(1:7)]
train_data_cleaned <- train_data_cleaned[, -grep("^kurtosis", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^skewness_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^min_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^max_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^var_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^avg_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^stddev_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^total_", names(train_data_cleaned))]
train_data_cleaned <- train_data_cleaned[, -grep("^amplitude_", names(train_data_cleaned))]
```

Now that the dataset has been reduced a lot through logical operations we can see if any columns can be removed due to having low variance.

``` {r variance_cleaning, cache = TRUE}
nrzv <- nearZeroVar(train_data_cleaned)
nrzv
```
There's nothing with low variation to be removed.

## Model Training

Now that the dtaa has been reduced, and there were then no near zero variance columns to be removed, we can try and build a model. The model chosen is a Random Forest model, with train controls utilising cross validation in the form of k-fold. The number of k-folds isn't particularly high as we have already set aside a subset of data for use in validation.

``` {r model_build, cache = TRUE, warning = FALSE}
control <- trainControl(method = "cv", number = 3, allowParallel = TRUE)
model <- train(classe ~ ., data = train_data_cleaned, method="rf", trControl = control)
```


### Model Testing

``` {r training_prediciton, cache = TRUE, warning = FALSE}
train_predict <- predict(model, train_data_cleaned)
confusionMatrix(train_predict, train_data_cleaned$classe)
```

Unsurprisingly a very good performance against the training set. Now to check it against the validation set.

``` {r validation_prediciton, cache = TRUE, warning = FALSE}
validation_predict <- predict(model, validation_data)
confusionMatrix(validation_predict, validation_data$classe)
```

So an expected out of sample accuracy of 99.2%, and an out of sample error rate of 0.8%.

