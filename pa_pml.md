---
title: "PA PML"
author: "Pascal P"
date: "8 September 2018"
output: 
  html_document:
    toc: true
    toc_depth: 3
    number_sections: true
    keep_md: true
    df_print: paged
    highlight: zenburn
    theme: simplex

---



# Synopsis
The aim of this report is to predict the manner in which a group of six people did their fitness exercise, using data from accelerometers on the belt, forearm, arm, and dumbell. More information about the datasets (a training set and a test one) is available at http://groupware.les.inf.puc-rio.br/har (in particular cf. section Weight Lifting Exercises Dataset).

Our procedure involve the following steps in this order:
  
- load the packages we need,
- load the two datasets,
- clean up these two datasets in order the reduce the dimension without loosing on our prediction objective,
- split the training dataset into two subsets: a training one and a validation one,
- define three "simple" models: decision tree (CART), LDA and SVM for predicting the outcome variable `classe`,
- use bagging technique with bagged CART (Classification and Regression Tree) and RF (Random Forest),
- use boosting technique with SGB (Stochastic Gradient Boosting), using SBM model
- use stacking technique combining predictions from CART, LDA  and SVM (*moved to Appendix*),
- use the best model(s) to make prediction for the outcome variable(`classe`) with the test data set.

In each case we will present the accuracy and Kappa on training and validation sets plus the out-of sample error (abbreviated as `oose` in the rest of this document). A complete comparison is provided in the "Summary" section, while the appendix provide even more details about the intermediate results.

The whole code for this project was put into a `R script` file named `pml.R` available in this repository: https://github.com/pascal-p/PA_PML 


# Data Processing

## Loading required packages and setting the seed

```r
library(caret)
# library(caretEnsemble)
library(rpart)          # for decision tree
library(randomForest)

set.seed(20180905)
```

## Loading the data

For this project we downloaded the two datasets from:

- training data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
- testing data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


Note: the original datasets are as follows:

- training dataset of dimension $19622\times160$
- a test one of dimension $20\times160$


```r
if (!exists("df_training")) { df_training <- loadDS() }               # df_training == 19622x160
if (!exists("df_testing")) { df_testing  <- loadDS(dset='testing') }  # df_testing  == 20x160
```

## Cleaning up the data

We use the following three step process:

1. Remove variables with near zero variance (low contribution in explaining our outcome variable)
1. Remove variables with missing values
1. Finally the first six columns of our dataset which are ids, name and timestamps can be removed as these are very unlikely to determine the outcome (`classe` variable).


```r
cleanDS <- function(df_training, df_testing, col2rm=c(1:6)) {
  # 1 - eliminate near zero variance variables
  res <- nearZeroVar(df_training, saveMetrics=TRUE)
  df_train_reduce <- df_training[ , !res$nzv] # 19622 x 117
  df_test_reduce  <- df_testing[ , !res$nzv]  #    20 x 117
  rm(list=c("df_training", "df_testing"))
  
  # 2 - eliminate variables (columns) containing NA values
  cols_wo_na <- colSums(is.na(df_train_reduce)) == 0
  df_train_reduce <- df_train_reduce[which(cols_wo_na)] # 19622 x 59
  df_test_reduce  <- df_test_reduce[which(cols_wo_na)]  #    20 x 59
  
  # 3 - Finally the first 6 colnames: ids, name and timestamps can be removed
  #   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp num_window
  df_training <- df_train_reduce[, -col2rm] # 19622x53
  df_testing <- df_test_reduce[, -col2rm]   #  20x53
  
  return(list(df_training, df_testing))
}

# clean
all_ds <- cleanDS(df_training, df_testing)
df_trainingC <- all_ds[[1]] # 19622x53
df_testing  <- all_ds[[2]]  # 20x53
rm(all_ds)
```

Our datasets have now the following dimensions:

- training dataset, dimension: $19622\times53$
- test set, dimension: $20\times53$


## Splitting the data

We split our training dataset into two subsets:

- training (70%) and
- validation (30%, it will be used to define oose)


```r
splitDS <- function(df, cut=0.7) {
  mySet <- createDataPartition(df$classe, p=cut, list=F)
  trainSet <- df[mySet, ] 
  validSet <- df[-mySet, ]
  return(list(trainSet, validSet))
}

all_ds <- splitDS(df_trainingC)
df_trainSet <- all_ds[[1]] # 13737x53
df_validSet <- all_ds[[2]] # 5885x53
rm(all_ds)
```

## First models

Here we use the `caret` package to define three models: `CART`, `LDA` and `SVM`.

- Decision trees (CART) are robust to errors, missing values (here we eliminated them) and outliers and they are well suited for target function with discrete values (our case here), however they can overfit.

- LDA [Linear Discriminant Analysis] is a simple and effective method for classification that makes assumptions on the underlying data such as: the samples are drawn from a multivariate Gaussian (aka multivariate normal) with a class specific mean vector and a common covariance matrix. This may not be the case for our datasets.

- SVM [Support Vector Machine] does not make strong assumptions on the data, nor does it overfit. It does look like a very good candidate for our analysis.

For all of these algorithms  we use k-fold cross validation technique with ($k=10$).  

Check `Summary` section for a whole comparison between models (in term of *accuracy, Kappa and out-of sample errors*) and for the meaning of the column names.


```r
trCtrl <- function(meth='cv', k=10) {
  return(trainControl(method=meth, number=k,
                      verboseIter=F, savePredictions=F))
}
```



### CART model


```r
modCARTfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  model <- train(classe ~ ., data=trainSet, method="rpart", 
                 trControl=trctrl)
  predRes(model, validSet)
}
```


Table: CART Summary

ds.names    ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---------  ----------  ------------  ----------  ------------  ----------  ---------
CART        0.5009162      0.348063   0.4941376      0.338296   0.5058624     12.159


Remark: 

- the performance is not very impressive.


### LDA model


```r
modLDAfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  model <- train(classe ~ ., data=trainSet, method="lda", 
                 trControl=trctrl)
  predRes(model, validSet)
}
```


Table: LDA Summary

     ds.names    ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---  ---------  ----------  ------------  ----------  ------------  ----------  ---------
2    LDA         0.7037169     0.6250863   0.6973662     0.6172881   0.3026338       6.59


Remark: 

- here we can see a definitive improvement of the performance of LDA over the previous attempt with CART.

### SVM model


```r
modSVMfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  model <- train(classe ~ ., data=trainSet, method="svmRadial", 
                 trControl=trctrl, prox=F)  
  predRes(model, validSet) 
}
```

Table: SVM Summary

     ds.names    ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs    ds.oose   ds.utime
---  ---------  ----------  ------------  ----------  ------------  ---------  ---------
3    SVM         0.9232702     0.9027418    0.919966     0.8985517   0.080034   1827.863

Remarks: 

- we can see a definitive improvement in the performance of SVM over the previous attempts with LDA and CART.
- the procedure is more involved and CPU intensive though (over 30 minutes against approximately 10 seconds in the previous cases).

## Bagging technique

Now we are going to define several bagging techniques in an attempt improve on the previous methods (this is called ensemble predictions):

- Bagging CART,
- Bagging LDA,
- Random Forest (an extension of bagging, using "feature" bagging).

Remarks: 

- I failed to get results with bagging SVM. I gave up after over 12 hours run without results (more advance techniques) and failures with less involved techniques, more investigation is required.  
- I Did not try to fine tune these models.

### Bagging CART


```r
modBA_CARTfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  model <- train(classe ~ ., data=trainSet, method="treebag", 
                 trControl=trctrl)
  predRes(model, validSet)
}
```


Table: Summary Bagging CART

     ds.names        ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---  -------------  ----------  ------------  ----------  ------------  ----------  ---------
4    Bagging CART    0.9855152     0.9816769   0.9828377     0.9782894   0.0171623    209.319

Remarks:

- the improvement over the single CART model is impressive with 98.3% on validation set (it was 49.4% for CART), so close to a factor 2,
- the `oose` dropped from 50.6% to 1.7%.
- the processing time increased by a factor of 19 for bagged LDA model.

### Bagging LDA


```r
modBA_LDAfn <- function(trainSet, validSet) {
  bagCtrl <- bagControl(fit=ldaBag$fit, predict=ldaBag$pred, aggregate=ldaBag$aggregate)
  # no tuning
  model <- train(classe ~ ., data=trainSet, method="bag",
                 B=50, allowParallel=T,
                 bagControl=bagCtrl)
  predRes(model, validSet)
}
```


Table: Bagging LDA Summary

     ds.names       ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---  ------------  ----------  ------------  ----------  ------------  ----------  ---------
5    Bagging LDA    0.7029541     0.6241709   0.6961767     0.6156871   0.3038233    897.269

Remark: 

- no improvement nor any deterioration in this case for the results (are the samples too similar?).  


### Random Forest


```r
modRFfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  model <- train(classe ~ ., data=trainSet, method="rf", 
                  trControl=trctrl, prox=F)
  predRes(model, validSet) 
}
```


Table: RF Summary

     ds.names         ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---  --------------  ----------  ------------  ----------  ------------  ----------  ---------
6    Random Forest    0.9917026     0.9895035   0.9920136     0.9898954   0.0079864   1435.185


Remark: 

- this model is even better than Bagging CART, it even performs slightly better on validation set with 99.2% of accuracy (compare to training set).

## Boosting technique

Finally, we picked one algorithm from another family of ensemble techniques called boosting, namely SGB (Stochastic Gradient Boosting) using the Gradient Boosting Modeling as defined in the caret package.

### GBM


```r
modGBMfn<- function(trainSet, validSet, trctrl=trCtrl(), metric="Accuracy") {
  model <- train(classe ~ ., data=trainSet, method="gbm", 
                 trControl=trctrl, metric=metric, verbose=F)
  predRes(model, validSet) 
}
```


Table: GBM Summary

     ds.names    ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---  ---------  ----------  ------------  ----------  ------------  ----------  ---------
7    GBM         0.9628018     0.9529358   0.9634664     0.9537914   0.0365336    518.505

Remark: 

- this model yields good results and ranks at the third place behind Random Forest and Bagging CART models.  

## Stacking technique

Cf. appendix for all the details.

# Summary

We now present the summary for all the models created in the previous sections.  

The columns are named as follows:

- `ds.names` - name of the algorithm technique used,
- `ds.acc_tr`, `ds_acc_kappa_tr` - the Accuracy and Kappa statistic on the training set,
- `ds.acc_vs`, `ds_acc_kappa_vs` - the Accuracy and Kappa statistic on the validation set,
- `ds.oose` - the out-of sample error (calculated on validation set),
- `ds.utime` - the user time taken by the algorithm (on training set using cross-validation) expressed in seconds.



Table: Summary of models

ds.names         ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
--------------  ----------  ------------  ----------  ------------  ----------  ---------
CART             0.5009162     0.3480630   0.4941376     0.3382960   0.5058624     12.159
LDA              0.7037169     0.6250863   0.6973662     0.6172881   0.3026338      6.590
SVM              0.9232702     0.9027418   0.9199660     0.8985517   0.0800340   1827.863
Bagging CART     0.9855152     0.9816769   0.9828377     0.9782894   0.0171623    209.319
Bagging LDA      0.7029541     0.6241709   0.6961767     0.6156871   0.3038233    897.269
Random Forest    0.9917026     0.9895035   0.9920136     0.9898954   0.0079864   1435.185
GBM              0.9628018     0.9529358   0.9634664     0.9537914   0.0365336    518.505

The three best models in decreasing order (of accuracy on the validation set) are:

1. Bagging Random Forest,
2. Bagging CART,
3. Boosting with GBM.


# Prediction and conclusion

Let's run our best predictors on the test set.

- With Bagging Random Forest:

```
## [1] "From BA RF: "
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

- With Bagging CART:

```
## [1] "From BA CART: "
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

- With Boosting GBM:

```
## [1] "From GBM: "
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

Our three models agree on the prediction. The next step could be to stack them together in a meta-model and see how it would behave on a bigger test set.

This work could be continued by applying:

- other algorithms and/or
- invest time on the tuning of the algorithm used (each of them offer some set of options to go further).

Finally, I learned a few techniques while working on this project *and not surprisingly* it is also clear to me that to master these algorithms, methods in their details it will take a long time (which path to mastery is not a long journey?).  


# Appendix 

Here, I am providing some more details about the results.  
For the code source, once again please refer to R script `pml.R`(available here: https://github.com/pascal-p/PA_PML)

## First models

### CART details


```
CART 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 12364, 12364, 12361, 12364, 12365, 12363, ... 
Resampling results across tuning parameters:

  cp       Accuracy  Kappa  
  0.03784  0.5009    0.34806
  0.05991  0.4285    0.22981
  0.11586  0.3326    0.07351

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was cp = 0.03784.
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1520   24  126    0    4
         B  480  382  277    0    0
         C  474   28  524    0    0
         D  454  176  334    0    0
         E  167  150  283    0  482

Overall Statistics
                                         
               Accuracy : 0.4941         
                 95% CI : (0.4813, 0.507)
    No Information Rate : 0.5259         
    P-Value [Acc > NIR] : 1              
                                         
                  Kappa : 0.3383         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.4911  0.50263  0.33938       NA  0.99177
Specificity            0.9448  0.85229  0.88436   0.8362  0.88887
Pos Pred Value         0.9080  0.33538  0.51072       NA  0.44547
Neg Pred Value         0.6260  0.92035  0.79008       NA  0.99917
Prevalence             0.5259  0.12914  0.26236   0.0000  0.08258
Detection Rate         0.2583  0.06491  0.08904   0.0000  0.08190
Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
Balanced Accuracy      0.7180  0.67746  0.61187       NA  0.94032
```

### LDA details

```
Linear Discriminant Analysis 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 12364, 12364, 12361, 12364, 12365, 12363, ... 
Resampling results:

  Accuracy  Kappa 
  0.7037    0.6251
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1358   37  142  133    4
         B  167  738  117   60   57
         C  101   99  649  149   28
         D   57   37  116  721   33
         E   37  192   98  117  638

Overall Statistics
                                          
               Accuracy : 0.6974          
                 95% CI : (0.6854, 0.7091)
    No Information Rate : 0.2923          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6173          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.7895   0.6691   0.5784   0.6110   0.8395
Specificity            0.9241   0.9161   0.9208   0.9484   0.9134
Pos Pred Value         0.8112   0.6479   0.6326   0.7479   0.5896
Neg Pred Value         0.9140   0.9231   0.9027   0.9067   0.9746
Prevalence             0.2923   0.1874   0.1907   0.2005   0.1291
Detection Rate         0.2308   0.1254   0.1103   0.1225   0.1084
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.8568   0.7926   0.7496   0.7797   0.8764
```

### SVM details


```
Support Vector Machines with Radial Basis Function Kernel 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 12364, 12364, 12361, 12364, 12365, 12363, ... 
Resampling results across tuning parameters:

  C     Accuracy  Kappa 
  0.25  0.8639    0.8275
  0.50  0.8951    0.8670
  1.00  0.9233    0.9027

Tuning parameter 'sigma' was held constant at a value of 0.01379938
Accuracy was used to select the optimal model using the largest value.
The final values used for the model were sigma = 0.0138 and C = 1.
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1656    6   11    1    0
         B  113  972   46    3    5
         C    6   51  961    8    0
         D    6    6  112  839    1
         E    0   12   42   42  986

Overall Statistics
                                          
               Accuracy : 0.92            
                 95% CI : (0.9127, 0.9268)
    No Information Rate : 0.3026          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.8986          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9298   0.9284   0.8200   0.9395   0.9940
Specificity            0.9956   0.9655   0.9862   0.9750   0.9804
Pos Pred Value         0.9892   0.8534   0.9366   0.8703   0.9113
Neg Pred Value         0.9703   0.9842   0.9566   0.9890   0.9988
Prevalence             0.3026   0.1779   0.1992   0.1517   0.1686
Detection Rate         0.2814   0.1652   0.1633   0.1426   0.1675
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9627   0.9469   0.9031   0.9572   0.9872
```

## Bagging


```
Bagged CART 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 12364, 12364, 12361, 12364, 12365, 12363, ... 
Resampling results:

  Accuracy  Kappa 
  0.9855    0.9817
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1666    3    4    1    0
         B   17 1100   15    6    1
         C    1    5 1012    8    0
         D    1    2   17  942    2
         E    0    4    4   10 1064

Overall Statistics
                                         
               Accuracy : 0.9828         
                 95% CI : (0.9792, 0.986)
    No Information Rate : 0.2863         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9783         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9887   0.9874   0.9620   0.9741   0.9972
Specificity            0.9981   0.9918   0.9971   0.9955   0.9963
Pos Pred Value         0.9952   0.9658   0.9864   0.9772   0.9834
Neg Pred Value         0.9955   0.9971   0.9918   0.9949   0.9994
Prevalence             0.2863   0.1893   0.1788   0.1643   0.1813
Detection Rate         0.2831   0.1869   0.1720   0.1601   0.1808
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9934   0.9896   0.9795   0.9848   0.9967
```

### LDA details

```
Bagged Model 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
Resampling results:

  Accuracy  Kappa 
  0.703     0.6242

Tuning parameter 'vars' was held constant at a value of 52
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1359   35  143  133    4
         B  170  737  119   58   55
         C  107   99  645  148   27
         D   59   36  119  718   32
         E   37  188  100  119  638

Overall Statistics
                                          
               Accuracy : 0.6962          
                 95% CI : (0.6842, 0.7079)
    No Information Rate : 0.2943          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6157          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.7846   0.6731   0.5728   0.6105   0.8439
Specificity            0.9242   0.9161   0.9199   0.9478   0.9134
Pos Pred Value         0.8118   0.6471   0.6287   0.7448   0.5896
Neg Pred Value         0.9114   0.9246   0.9010   0.9069   0.9754
Prevalence             0.2943   0.1861   0.1913   0.1998   0.1285
Detection Rate         0.2309   0.1252   0.1096   0.1220   0.1084
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.8544   0.7946   0.7464   0.7792   0.8787
```

### RF details


```
Random Forest 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 12364, 12364, 12361, 12364, 12365, 12363, ... 
Resampling results across tuning parameters:

  mtry  Accuracy  Kappa 
   2    0.9917    0.9895
  27    0.9916    0.9894
  52    0.9872    0.9838

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was mtry = 2.
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    1    0    0    0
         B   13 1123    3    0    0
         C    0   11 1015    0    0
         D    0    0   15  949    0
         E    0    0    0    4 1078

Overall Statistics
                                          
               Accuracy : 0.992           
                 95% CI : (0.9894, 0.9941)
    No Information Rate : 0.2865          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9899          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9923   0.9894   0.9826   0.9958   1.0000
Specificity            0.9998   0.9966   0.9977   0.9970   0.9992
Pos Pred Value         0.9994   0.9860   0.9893   0.9844   0.9963
Neg Pred Value         0.9969   0.9975   0.9963   0.9992   1.0000
Prevalence             0.2865   0.1929   0.1755   0.1619   0.1832
Detection Rate         0.2843   0.1908   0.1725   0.1613   0.1832
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9960   0.9930   0.9902   0.9964   0.9996
```

## Boosting - GBM


```
Stochastic Gradient Boosting 

13737 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 12364, 12364, 12361, 12364, 12365, 12363, ... 
Resampling results across tuning parameters:

  interaction.depth  n.trees  Accuracy  Kappa 
  1                   50      0.7484    0.6809
  1                  100      0.8190    0.7710
  1                  150      0.8549    0.8164
  2                   50      0.8555    0.8169
  2                  100      0.9082    0.8838
  2                  150      0.9333    0.9156
  3                   50      0.8976    0.8703
  3                  100      0.9431    0.9280
  3                  150      0.9628    0.9529

Tuning parameter 'shrinkage' was held constant at a value of 0.1

Tuning parameter 'n.minobsinnode' was held constant at a value of 10
Accuracy was used to select the optimal model using the largest value.
The final values used for the model were n.trees = 150,
 interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1639   25    6    3    1
         B   33 1080   24    1    1
         C    0   27  990    9    0
         D    0    3   32  923    6
         E    1   16    6   21 1038

Overall Statistics
                                          
               Accuracy : 0.9635          
                 95% CI : (0.9584, 0.9681)
    No Information Rate : 0.2843          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9538          
 Mcnemar's Test P-Value : 1.269e-07       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9797   0.9383   0.9357   0.9645   0.9924
Specificity            0.9917   0.9875   0.9925   0.9917   0.9909
Pos Pred Value         0.9791   0.9482   0.9649   0.9575   0.9593
Neg Pred Value         0.9919   0.9850   0.9860   0.9931   0.9983
Prevalence             0.2843   0.1956   0.1798   0.1626   0.1777
Detection Rate         0.2785   0.1835   0.1682   0.1568   0.1764
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9857   0.9629   0.9641   0.9781   0.9916
```

## Stacking technique

- We first need to define three different sets: training, testing and validation


```
## [1] 11776    53
## [1] 3923   53
## [1] 3923   53
```

- We then build three models as before on the (smaller) training set: CART, LDA and SVM.




Table: Summary of models

ds.names    ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
---------  ----------  ------------  ----------  ------------  ----------  ---------
CART        0.5195155     0.3884948   0.5569717     0.4461376   0.4430283      9.099
LDA         0.7020224     0.6230138   0.7004843     0.6206273   0.2995157      4.960
SVM         0.9150858     0.8923679   0.9153709     0.8926851   0.0846291   1170.148

- We combine the previous models using Random Forest in similar fashion as what was done in the course (week 4 - combining predictors):


Table: Summary of models - with stacking technique

ds.names       ds.acc_tr   ds.kappa_tr   ds.acc_vs   ds.kappa_vs     ds.oose   ds.utime
------------  ----------  ------------  ----------  ------------  ----------  ---------
CART           0.5195155     0.3884948   0.5569717     0.4461376   0.4430283      9.099
LDA            0.7020224     0.6230138   0.7004843     0.6206273   0.2995157      4.960
SVM            0.9150858     0.8923679   0.9153709     0.8926851   0.0846291   1170.148
Stacking RF    0.9144693     0.8915510   0.9189396     0.8972036   0.0810604    111.791

Remark:

- Stacking barely improves over the best single models (SVM). The accuracy is slightly better on the validation set 91.8%.

# References

(main) Websites:

- Coursera ["Practical Machine Learning" course](https://www.coursera.org/learn/practical-machine-learning/home/welcome)
- *Weight Lifting Exercise Dataset from groupware*,  http://groupware.les.inf.puc-rio.br/har
- Machine Learning Mastery - https://machinelearningmastery.com/ and in particular this reference:  
  https://machinelearningmastery.com/machine-learning-ensembles-with-r/

(main) Books:

- *An Introduction to Statistical Learning: with Applications in R*, by Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani,
- *The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition* by Trevor Hastie and Robert Tibshirani,
- *R in Action: Data Analysis and Graphics with R Second Edition* by Robert Kabacoff



