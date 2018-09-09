#
# (cl) Pascal - 2018
#
# THIS SCRIPT IS PROVIDED BY THE COPYLEFT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYLEFT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SCRIPT, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#


library(caret)
# library(caretEnsemble)

library(rpart)          # for decision tree

library(randomForest)
library(kernlab)         # for SVM



setSeed <- function(s=20180905) {
  set.seed(s)
}

setDataDir <- function(dir="data") {
  (!file.exists(dir)) && { dir.create(dir) }
  return(dir)
}

loadDS <- function (url_prefx='https://d396qusza40orc.cloudfront.net/predmachlearn', dset="training",
                    prefx='pml', suffx='csv', dir=setDataDir()) {

  file <- paste(prefx, dset, sep='-')
  file <- paste(file, suffx, sep='.')
  fpath <- paste(dir, file, sep='/')

  if (!file.exists(fpath)) {
    download.file(url=paste(url_prefx, file, sep='/'), method="curl", destfile=fpath)
  }

  df <- read.csv(fpath, na.strings=c('NA', ''), header=T)
  return(df)
}

cleanDS <- function(df_training, df_testing, col2rm=c(1:6)) {
  # 1 - eliminate variable with low variance
  res <- nearZeroVar(df_training, saveMetrics=TRUE)
  df_train_reduce <- df_training[ , !res$nzv] # 19622 x 117
  df_test_reduce  <- df_testing[ , !res$nzv]  #    20 x 117
  rm(list=c("df_training", "df_testing"))

  # 2 - eliminate column containin NA values - decision trees not affected by that
  cols_wo_na <- colSums(is.na(df_train_reduce)) == 0
  df_train_reduce <- df_train_reduce[which(cols_wo_na)] # 19622 x 59
  df_test_reduce  <- df_test_reduce[which(cols_wo_na)]  #    20 x 59

  # 3 - finally the following (first 6 colnames from dataframe): ids, name and timestamps are unlikely to determine our outcome 'classe'
  # X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp num_window
  df_training <- df_train_reduce[, -col2rm] # 19622x53
  df_testing <- df_test_reduce[, -col2rm]   #  20x53

  return(list(df_training, df_testing))
}

splitDS <- function(df, cut=0.7) {
  mySet <- createDataPartition(df$classe, p=cut, list=F)
  trainSet <- df[mySet, ]
  validSet <- df[-mySet, ]
  return(list(trainSet, validSet))
}

trCtrl <- function(meth='cv', k=10) {
  return(trainControl(method=meth, number=k,
                      verboseIter=F, savePredictions=F))
}

trCtrlR <- function(meth='repeatedcv', k=10, rep=3) {
  return(trainControl(method=meth, number=k, repeats=rep, # classProbs=T,
                      verboseIter=F, savePredictions='final'))
}

predRes <- function(model, validSet) {
  pred <- predict(model, newdata=validSet)
  # Eval out-of-sample-error == oose
  resCM <- confusionMatrix(validSet$classe, pred)
  oose <- 1 - resCM$overall[[1]]
  return(list(model, pred, resCM, oose))
}

## "Single" models
modCARTfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  setSeed()
  model <- train(classe ~ ., data=trainSet, method="rpart",
                 trControl=trctrl)
  predRes(model, validSet)
}

modLDAfn <-  function(trainSet, validSet, trctrl=trCtrl()) {
  setSeed()
  model <- train(classe ~ ., data=trainSet, method="lda",
                 trControl=trctrl)
  predRes(model, validSet)
}

modSVMfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  setSeed()
  model <- train(classe ~ ., data=trainSet, method="svmRadial",
                 trControl=trctrl, prox=F)
  predRes(model, validSet)
}

## Bagging
modBA_CARTfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  setSeed()
  model <- train(classe ~ ., data=trainSet, method="treebag",
                 trControl=trctrl)
  predRes(model, validSet)
}

modBA_LDAfn <- function(trainSet, validSet) {
  setSeed()
  bagCtrl <- bagControl(fit=ldaBag$fit, predict=ldaBag$pred, aggregate=ldaBag$aggregate)
  # no tuning
  model <- train(classe ~ ., data=trainSet, method="bag",
                 B=50, allowParallel=T,
                 bagControl=bagCtrl)
  predRes(model, validSet)
}

modBA_SVMfn <- function(trainSet, validSet) {
  bagCtrl <- bagControl(fit=svmBag$fit, predict=svmBag$pred, aggregate=svmBag$aggregate)
  model <- train(classe ~ ., data=trainSet, method="bag",
                 B=100, allowParallel=T, bagControl=bagCtrl)
  predRes(model, validSet)
}

modRFfn <- function(trainSet, validSet, trctrl=trCtrl()) {
  setSeed()
  model <- train(classe ~ ., data=trainSet, method="rf",
                  trControl=trctrl, prox=F)
  predRes(model, validSet)
}

## Boosting
modGBMfn<- function(trainSet, validSet, trctrl=trCtrl()) {
  setSeed()
  model <- train(classe ~ ., data=trainSet, method="gbm",
                 trControl=trctrl, verbose=F)
  predRes(model, validSet)
}

## CaretEnsemble

## modSelect <- function(trainSet, validSet, trctrl=trCtrlR(),
##                       algos=c('lda', 'rpart', 'svmRadial') ) {
##   setSeed()
##   # return a list of models
##   models <- caretList(classe ~ ., data=trainSet, trControl=trctrl,
##                       methodList=algos)
##   for (algo in algos) {
##     pred <- predict(models[algo], validSet)
##   }
## }

## error in check_caretList_model_types(list_of_models) :
##   Not yet implemented for multiclass problems
## modStack <- function(models, validSet, stackctrl=trCtrlR(),
##                      metric="Accuracy", meta_meth="glm") {
##   stack <- caretStack(models, method=meta_meth, metric=metric, trControl=stackctrl)
##   # TODO:  use ValidSet ...
##   return(stack)
## }

modStackfn <- function(models, validSet, ...) {
  ## TODO
}

resDF <- function(df=NULL, model=model, label=label, resCM=res, oose=oose) {
  ix_max      <- which.max(model$results$Accuracy)
  ds.names    <- c(label) # "DT"
  ds.acc_tr   <- c(model$results$Accuracy[[ix_max]])
  ds.kappa_tr <- c(model$results$Kappa[[ix_max]])
  ds.acc_vs   <- c(resCM$overall[[1]])
  ds.kappa_vs <- c(resCM$overall[[2]])
  ds.oose     <- c(oose)
  ds.utime    <- model$times$everything[[1]]

  ndf <- data.frame(ds.names, ds.acc_tr, ds.kappa_tr, ds.acc_vs, ds.kappa_vs, ds.oose, ds.utime)
  if (exists("df")) { rbind(df, ndf) }
  else { ndf }
}





##
## Main
##

setSeed()
if (!exists("df_training")) { df_training <- loadDS() }              # df_training == 19622x160
if (!exists("df_testing")) { df_testing  <- loadDS(dset='testing') } # df_testing  == 20x160

## clean
if (!exists("df_trainingC")) {
  all_ds <- cleanDS(df_training, df_testing)
  df_trainingC <- all_ds[[1]]
  df_testing  <- all_ds[[2]]
  rm(all_ds)
}

## split
if (!exists("df_trainSet") || !exists("df_validSet")) {
  all_ds <- splitDS(df_trainingC)
  df_trainSet <- all_ds[[1]]
  df_validSet <- all_ds[[2]]
  rm(all_ds)
}

## 1a - CART / Dec Tree

if (!exists("modCART")) {
  ix <- 1
  print(" model CART... Be patient...")
  all <- modCARTfn(df_trainSet, df_validSet)
  modCART  <- all[[1]]
  predCART <- all[[2]]
  resCART  <- all[[3]]
  ooseCART <- all[[4]]
  sum_df <- resDF(model=modCART, label="CART", resCM=resCART, oose=ooseCART) # 1st...
  rm(all)
  print(sum_df[ix, ])
}

## 1b - LDA
if (!exists("modLDA")) {
  print(" modelLDA... Be patient...")
  all <- modLDAfn(df_trainSet, df_validSet)
  modLDA  <- all[[1]]
  predLDA <- all[[2]]
  resLDA  <- all[[3]]
  ooseLDA <- all[[4]]
  sum_df <- resDF(sum_df, model=modLDA, label="LDA", resCM=resLDA, oose=ooseLDA)
  rm(all)
  ix <- ix + 1
  print(sum_df[ix, ])
}

## 1c - SVM
if (!exists("modSVM")) {
  print(" model SVM... Be patient...")
  all <- modSVMfn(df_trainSet, df_validSet)
  modSVM  <- all[[1]]
  predSVM <- all[[2]]
  resSVM  <- all[[3]]
  ooseSVM <- all[[4]]
  sum_df <- resDF(sum_df, model=modSVM, label="SVM", resCM=resSVM, oose=ooseSVM)
  ix <- ix + 1
  print(sum_df[ix, ])
}

## 2a - Bagging CART
if (!exists("modBA_CART")) {
  print(" model bagging CART... Be patient...")
  all <- modBA_CARTfn(df_trainSet, df_validSet)
  modBA_CART  <- all[[1]]
  predBA_CART <- all[[2]]
  resBA_CART  <- all[[3]]
  ooseBA_CART <- all[[4]]
  sum_df <- resDF(sum_df, model=modBA_CART, label="Bagging CART", resCM=resBA_CART, oose=ooseBA_CART)
  rm(all)
  ix <- ix + 1
  print(sum_df[ix, ])
}

## 3b - Bagging LDA
if (!exists("modBA_LDA")) {
  print(" model bagging LDA... Be patient...")
  all <- modBA_LDAfn(df_trainSet, df_validSet)
  modBA_LDA  <- all[[1]]
  predBA_LDA <- all[[2]]
  resBA_LDA  <- all[[3]]
  ooseBA_LDA <- all[[4]]
  sum_df <- resDF(sum_df, model=modBA_LDA, label="Bagging LDA", resCM=resBA_LDA, oose=ooseBA_LDA)
  rm(all)
  ix <- ix + 1
  print(sum_df[ix, ])
}

## 3c - Bagging SVM
## if (!exists("modBA_SVM")) {
##   print(" model bagging SVM... Be patient...")
##   all <- modBA_SVMfn(df_trainSet, df_validSet)
##   modBA_SVM  <- all[[1]]
##   predBA_SVM <- all[[2]]
##   resBA_SVM  <- all[[3]]
##   ooseBA_SVM <- all[[4]]
##   sum_df <- resDF(sum_df, model=modBA_SVM, label="Bagging SVM", resCM=resBA_SVM, oose=ooseBA_SVM)
##   rm(all)
##   ix <- ix + 1
##   print(sum_df[ix, ])
## }

## 3d - Bagging Random Forest
if (!exists("modRF")) {
  print(" model bagging RF... Be patient...")
  all <- modRFfn(df_trainSet, df_validSet)
  modRF  <- all[[1]]
  predRF <- all[[2]]
  resRF  <- all[[3]]
  ooseRF <- all[[4]]
  sum_df <- resDF(sum_df, model=modRF, label="Random Forest", resCM=resRF, oose=ooseRF)
  rm(all)
  ix <- ix + 1
  print(sum_df[ix, ])
}

## 4 - Boosting with GBM [Stochastic Gradient Modeling]
if (!exists("modGBM")) {
  print(" model GBM... Be patient...")
  all <- modGBMfn(df_trainSet, df_validSet)
  modGBM  <- all[[1]]
  predGBM <- all[[2]]
  resGBM  <- all[[3]]
  ooseGBM <- all[[4]]
  sum_df <- resDF(sum_df, model=modGBM, label="GBM", resCM=resGBM, oose=ooseGBM)
  rm(all)
  ix <- ix + 1
  print(sum_df[ix, ])
}

print(sum_df)

## 5 - Stacking - using CART, LDA and SVM - but need df_nTrainSet, df_nTestSet, df_nValidSet
if ((!exists("df_ntrainSet")) || (!exists("df_ntestSet")) || (!exists("df_nvalidSet"))) {
  all_ds <- splitDS(df_trainingC, cut=0.6)
  df_ntrainSet <- all_ds[[1]]
  df_otherSet <- all_ds[[2]]

  all_ds <- splitDS(df_otherSet, cut=0.5)
  df_ntestSet <- all_ds[[1]]
  df_nvalidSet <- all_ds[[2]]
  rm(all_ds)
  rm(df_otherSet)
}

## print(dim(df_ntrainSet)) # 11776x53
## print(dim(df_ntestSet))  #  3923x53
## print(dim(df_nvalidSet)) #  3923x53

## 5.1 - build CART on smaller train set and eval. on test set
if (!exists("modnCART")) {
  all <- modCARTfn(df_ntrainSet, df_ntestSet)
  modnCART  <- all[[1]]
  prednCART <- all[[2]]
  resnCART  <- all[[3]]
  oosenCART <- all[[4]]
  sumn_df <- resDF(model=modnCART, label="nCART", resCM=resnCART, oose=oosenCART)
  rm(all)
  ix <- 1
  print(sumn_df[ix, ])
}

## 5.2 - build LDA on smaller train set and eval. on test set
if (!exists("modnLDA")) {
  print(" modelnLDA... Be patient...")
  all <- modLDAfn(df_ntrainSet, df_ntestSet)
  modnLDA  <- all[[1]]
  prednLDA <- all[[2]]
  resnLDA  <- all[[3]]
  oosenLDA <- all[[4]]
  sumn_df <- resDF(sumn_df, model=modnLDA, label="nLDA", resCM=resnLDA, oose=oosenLDA)
  rm(all)
  ix <- ix + 1
  print(sumn_df[ix, ])
}

## 5.3 - build SVM on smaller train set and eval. on test set
if (!exists("modnSVM")) {
  print(" model nSVM... Be patient...")
  all <- modSVMfn(df_ntrainSet, df_ntestSet)
  modnSVM  <- all[[1]]
  prednSVM <- all[[2]]
  resnSVM  <- all[[3]]
  oosenSVM <- all[[4]]
  sumn_df <- resDF(sumn_df, model=modnSVM, label="nSVM", resCM=resnSVM, oose=oosenSVM)
  ix <- ix + 1
  print(sumn_df[ix, ])
}

## > sumn_df
##  ds.names ds.acc_tr ds.kappa_tr ds.acc_vs ds.kappa_vs    ds.oose ds.utime
##1    nCART 0.5792321   0.4647232 0.5669131   0.4558068 0.43308692    9.695
##2     nLDA 0.6953968   0.6144717 0.7124650   0.6362462 0.28753505    5.206
##3     nSVM 0.9192404   0.8976439 0.9163905   0.8940155 0.08360948 1569.915


## 5.4 - build new dataset combining previous prediction 5.1 -> 5.3
if (!exists("modCombGAM")) {
  df_preds <- data.frame(prednCART, prednLDA, prednSVM, classe=df_ntestSet$classe)
  modCombGAM <- train(classe ~ ., data=df_preds, method="gam")
  predCombGAM <- predict(modCombGAM, newdata=df_preds)

  ## on validSet now
  prednCARTv <- predict(modnCART, newdata=df_nvalidSet)
  prednLDAv <-  predict(modnLDA, newdata=df_nvalidSet)
  prednSVMv <-  predict(modnSVM, newdata=df_nvalidSet)

  df_predsv <- data.frame(pred.CART=prednCARTv, pred.LDA=prednLDAv, pred.SVM=prednSVMv)
  predCombGAMv <- predict(modCombGAM, df_predsv)

  resCM.GAM <- confusionMatrix(df_nvalidSet$classe, predCombGAMv)
  oose.GAM  <- 1 - resCM.GAM$overall[[1]]

  sumn_df <- resDF(sumn_df, model=modCombGAM, label="Stacking GAM", resCM=resCM.GAM, oose=oose.GAM)

  ## > modCombGAM$results
  ##   select method  Accuracy     Kappa AccuracySD    KappaSD
  ## 1  FALSE GCV.Cp 0.4552111 0.3003985 0.01154664 0.01020776
  ## 2   TRUE GCV.Cp 0.4552111 0.3003985 0.01154664 0.01020776

##   Confusion Matrix and Statistics

##           Reference
## Prediction    A    B    C    D    E
##          A 1099   17    0    0    0
##          B   68  691    0    0    0
##          C    5  679    0    0    0
##          D    4  639    0    0    0
##          E    1  720    0    0    0

## Overall Statistics

##                Accuracy : 0.4563
##                  95% CI : (0.4406, 0.472)
##     No Information Rate : 0.7
##     P-Value [Acc > NIR] : 1

##                   Kappa : 0.3022
##  Mcnemar's Test P-Value : NA

## Statistics by Class:

##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9337   0.2516       NA       NA       NA
## Specificity            0.9938   0.9422   0.8256   0.8361   0.8162
## Pos Pred Value         0.9848   0.9104       NA       NA       NA
## Neg Pred Value         0.9722   0.3505       NA       NA       NA
## Prevalence             0.3000   0.7000   0.0000   0.0000   0.0000
## Detection Rate         0.2801   0.1761   0.0000   0.0000   0.0000
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9638   0.5969       NA       NA       NA


}

if (!exists("modCombRF")) {
  df_preds <- data.frame(prednCART, prednLDA, prednSVM, classe=df_ntestSet$classe)
  modCombRF <- train(classe ~ ., data=df_preds, method="rf")
  predCombRF <- predict(modCombRF, newdata=df_preds)

  ## on validSet now
  predmCARTv <- predict(modnCART, newdata=df_nvalidSet)
  predmLDAv <-  predict(modnLDA, newdata=df_nvalidSet)
  predmSVMv <-  predict(modnSVM, newdata=df_nvalidSet)

  df_predsv <- data.frame(pred.CART=predmCARTv, pred.LDA=predmLDAv, pred.SVM=predmSVMv)
  predCombRFv <- predict(modCombRF, df_predsv)

  resCM.RF <- confusionMatrix(df_nvalidSet$classe, predCombRFv)
  oose.RF  <- 1 - resCM.RF$overall[[1]]

  sumn_df <- resDF(sumn_df, model=modCombRF, label="Stacking RF", resCM=resCM.RF, oose=oose.RF)

## > modCombRF$results
##   mtry  Accuracy     Kappa  AccuracySD     KappaSD
## 1    2 0.9169967 0.8947420 0.004114343 0.005236491
## 2    7 0.9191257 0.8974795 0.004506842 0.005752663
## 3   12 0.9186548 0.8968870 0.005001858 0.006376091

## Confusion Matrix and Statistics

##           Reference
## Prediction    A    B    C    D    E
##          A 1099   10    7    0    0
##          B   65  650   42    0    2
##          C    5   34  633   11    1
##          D    4    2   58  577    2
##          E    1    9   38   13  660

## Overall Statistics

##                Accuracy : 0.9225
##                  95% CI : (0.9137, 0.9307)
##     No Information Rate : 0.2993
##     P-Value [Acc > NIR] : < 2.2e-16

##                   Kappa : 0.9018
##  Mcnemar's Test P-Value : < 2.2e-16

## Statistics by Class:

##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9361   0.9220   0.8136   0.9601   0.9925
## Specificity            0.9938   0.9661   0.9838   0.9801   0.9813
## Pos Pred Value         0.9848   0.8564   0.9254   0.8974   0.9154
## Neg Pred Value         0.9733   0.9826   0.9552   0.9927   0.9984
## Prevalence             0.2993   0.1797   0.1983   0.1532   0.1695
## Detection Rate         0.2801   0.1657   0.1614   0.1471   0.1682
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9650   0.9441   0.8987   0.9701   0.9869
}


## update sumn_df

## > sumn_df
##       ds.names ds.acc_tr ds.kappa_tr ds.acc_vs ds.kappa_vs    ds.oose ds.utime
## 1        nCART 0.5792321   0.4647232 0.5669131   0.4558068 0.43308692    9.695
## 2         nLDA 0.6953968   0.6144717 0.7124650   0.6362462 0.28753505    5.206
## 3         nSVM 0.9192404   0.8976439 0.9163905   0.8940155 0.08360948 1569.915
## 4 Stacking GAM 0.4552111   0.3003985 0.4562835   0.3022322 0.54371654   23.279
## 5  Stacking RF 0.9191257   0.8974795 0.9225083   0.9018167 0.07749172  116.816


## 6 - Final prediction on df_testing with  Bagging CART, Random Forest and ...

if (!exists("fresPredBA_CART")) {
  fresPredBA_CART <- predict(modBA_CART, df_testing)

  print("From BA CART: ")
  print(as.character(fresPredBA_CART))

## >  print("From BA CART: ")
## [1] "From BA CART: "
## >   print(as.character(fresPredBA_CART))
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B"
## [20] "B"

}

if (!exists("fresPredRF")) {
  fresPredRF <- predict(modRF, df_testing)
  print("From BA RF: ")
  print(as.character(fresPredRF))

## > print("From BA RF: ")
## [1] "From BA RF: "
## >   print(as.character(fresPredRF))
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B"
## [20] "B"
}

## if (!exists("models")) {
##   print(" model selection... Be patient...")
##   models <- modSelect(df_trainSet)
## }

# print(models, digits=4)

## if (!exists("stack.glm")) {
##   print(" model stacking with glm... Be patient...")
##   stack.glm <- modStack(models, df_validSet)
## }

## if (!exists("stack.rf")) {
##   print(" model stacking with rf... Be patient...")
##   stack.rf <- modStack(models, df_validSet, meta_meth="rf")
## }
