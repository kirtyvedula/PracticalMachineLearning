---
title: "Machine Learning Course Project"
author: "Kirty Vedula"
date: "04/25/2015"
output: html_document
---


```r
setwd("/home/kirty/R/Coursera/MachineLearning")
```


```r
library(caret)
```

### Reading data from the files downloaded

```r
testBulk <- read.csv("pml-testing.csv",na.strings=c("NA",""))
trainBulk <- read.csv("pml-training.csv",na.strings=c("NA",""))
```

### Cleaning up the data 

```r
NAs <- apply(trainBulk,2,function(x) {sum(is.na(x))}) 
cleanTrain <- trainBulk[,which(NAs == 0)]
cleanTest <- testBulk[,which(NAs == 0)]
```


```r
require(caret)
set.seed(2103)
```


```r
trainIndex <- createDataPartition(y = cleanTrain$classe, p=0.7,list=FALSE)
trainSet <- cleanTrain[trainIndex,]
crossValidationSet <- cleanTrain[-trainIndex,]
```
## Removing variables that have time, or names in it

```r
removeIndex <- as.integer(c(1,2,3,4,5,6))
trainSet <- trainSet[,-removeIndex]
testSet <- cleanTest[,-removeIndex]
```

## Fitting the model with 10-fold cross-validation

```r
mytrControl = trainControl(method = "cv", number = 10)
modelFit <- train(trainSet$classe ~.,data = trainSet, method="rf", trControl = mytrControl)
modelFit
```

```
## Random Forest 
## 
## 13737 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 12363, 12365, 12364, 12362, 12363, 12362, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9957782  0.9946593  0.002136599  0.002703498
##   27    0.9978896  0.9973305  0.001796098  0.002272287
##   53    0.9957047  0.9945663  0.002460250  0.003112788
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
## Running the prediction algorithm on the cross-validation set

```r
predicted <- predict(modelFit, crossValidationSet)
SampleError <- sum(predicted == crossValidationSet$classe)/nrow(crossValidationSet)
SampleError
```

```
## [1] 0.9976211
```
### Testing using the test set

```r
answers <- predict(modelFit, testSet)
```
