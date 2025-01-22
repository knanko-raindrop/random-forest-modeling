# Random Forest Modeling
library(doParallel)
library(caret)
library(pdp)
set.seed(101)

# parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

# Prepare output files
fpOUT_imp <- "relativeImportance.txt"
rnames <- c("Interval","Foliation","Tree","obj","n","mtry","R2","RMSE","MAE"
            ,"RR_OP","CR_OP","KE_OP","D50_OP","Dmax_OP"
            ,"specie","D_BH","H_tree","A_tree","A_branch_all","L_branch_all","B_all","B_branch","B_leaf","B_branch/leaf","CPA"
            ,"N_voxel_local","C_voxel_local","T_voxel_local","N_branch_local","A_branch_local","N_voxel_branch1","C_voxel_branch1","T_voxel_branch1","A_branch1","H_branch1","RCP"
)
write.table(t(rnames), fpOUT_imp, append=F, quote=F, row.names=F, col.names=F, sep=",")

fpOUT_pdp <- "partialDependence.txt"
x100 <- paste("x", 1:51, sep="")
y100 <- paste("y", 1:51, sep="")
rnames <- c("Interval","Foliation","Tree","obj","var",x100,y100)
write.table(t(rnames), fpOUT_pdp, append=F, quote=F, row.names=F, col.names=F, sep=",")

cat("\n### start random forest modeling \n")

fn <- paste("dataset_02min.txt", sep="")
df_all <- read.csv(fn, header=T)
df_all$RI <-  as.factor(df_all$RI)
df_all$Tree <- as.factor(df_all$Tree)
df_all$Foliation <- as.factor(df_all$Foliation)
df_all$Location <-  as.factor(df_all$Location)
df_all$specie <-  as.factor(df_all$specie)

df_all$Location <- factor(df_all$Location, levels=c("Cedar1a", "Cedar1b", "Cedar1c", "Cedar2a", "Cedar2b", "Cypress1a", "Cypress1b", "Cypress1c", "Cypress2a", "Cypress2b", "Zelkova1a", "Zelkova1b", "Zelkova1c", "Zelkova2a", "Zelkova2b", "Birch1a", "Birch1b", "Birch2a", "Birch2b"))
df_all$specie <- factor(df_all$specie, levels=c("Cedar", "Cypress", "Birch", "Zelkova"))
df_all$Tree <- factor(df_all$Tree, levels=c("Cedar1", "Cedar2", "Cypress1", "Cypress2", "Zelkova1", "Zelkova2", "Birch1", "Birch2"))

# Start calculation
TR <- "all"
list_Leaf <- c('F','U')
interval <- "02"

for(LF in list_Leaf){

  df <- subset(df_all, Foliation == LF)

  varlist <- names(df)
  varLast <- ncol(df)

  for (obj in c('fDR', 'fSP', 'fFR', 'D50_DR', 'D50_SP')){
    
    cat(sprintf(" - calculating [%s, %s, %s, %s]\n", obj,LF,TR,interval))
    # Splitting training data and test data
    inTrain <- createDataPartition(y=df[,obj], p=.75, list=F)
    df_train <- df[inTrain,]
    df_test <- df[-inTrain,]

    # Prediction by Random Forest
    fitControl <- trainControl(
      method="repeatedcv", 
      number=10, 
      repeats=3,
      selectionFunction = "oneSE"
    )
    
    modelRF <- train(
      df_train[,15:varLast],
      df_train[,obj],
      method = "rf", 
      tuneLength = 8,
      preProcess = c('center', 'scale', 'YeoJohnson', 'zv', 'medianImpute'), 
      trControl = fitControl
    )

    performance <- postResample(pred = predict(modelRF, df_test), obs = df_test[,obj])
    RMSE <- performance[1]
    R2 <- performance[2]
    MAE <- performance[3]

    # Getting and outputting importance
    imp <- varImp(modelRF, scale=F)
    imp
    impOUT <- t(imp$importance)
    impOUT <- append(impOUT, MAE, after=0)
    impOUT <- append(impOUT, RMSE, after=0)
    impOUT <- append(impOUT, R2, after=0)
    impOUT <- append(impOUT, modelRF$finalModel$mtry, after=0)
    impOUT <- append(impOUT, nrow(df), after=0)
    impOUT <- append(impOUT, obj, after=0)
    impOUT <- append(impOUT, TR, after=0)
    impOUT <- append(impOUT, LF, after=0)
    impOUT <- append(impOUT, interval, after=0)

    write.table(t(impOUT), fpOUT_imp, append=T, quote=F, row.names=F, col.names=F, sep=",")
    
    # Getting and outputting pdp
    for(i in 15:varLast){
      prt <- partial(modelRF, pred.var=varlist[i])
      num <- 51 - nrow(prt)
      d1 <- c(interval, LF, TR, names(df[obj]), varlist[i])
      d1 <- append(d1, t(prt[1]))
      d1 <- append(d1, rep(c(""), times=num))
      d1 <- append(d1, t(prt[2]))
      d1 <- append(d1, rep(c(""), times=num))
      write.table(t(d1), fpOUT_pdp, append=T, quote=F, row.names=F, col.names=F, sep=",")
    }
  }
  cat("====================\n")
}
stopCluster(cl)
