library(readr)
library(readsparse)
library(readr)
library(dplyr)
library(stargazer)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)
library(tensor)

library(adabag)
library(caret)

packages = c("parallel","doParallel","doSNOW")
invisible(xfun::pkg_attach(packages))
train_X<- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/MADELON/madelon_train.data", 
                     col_names = FALSE)
train_Y<- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/MADELON/madelon_train.labels", 
                     col_names = FALSE)
test_X <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/MADELON/madelon_valid.data", 
                     col_names = FALSE)
test_Y <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/MADELON/madelon_valid.labels", 
                     col_names = FALSE)

train_X=train_X[,-501]
test_X=test_X[,-501]

x_mean=as.numeric(colMeans(train_X))
x_sd =as.numeric(apply(train_X,2,sd))

X = rbind(scale(train_X,center=x_mean,scale=x_sd),
          scale(test_X,center=x_mean,scale=x_sd))
X = X[, colSums(is.na(X)) == 0]
length(which(is.na(X) == TRUE))
# setting up train data
X_train = X[1:2000, ]
# dim(X_train)
data_train = list(y = as.matrix(train_Y), x = as.matrix(X_train))
# str(data_train)
#setting up test data
X_test = X[2001:2600, ]
# dim(X_test)
data_test = list(y = as.matrix(test_Y), x = as.matrix(X_test))
# str(data_test)
# Writing setup function
rm(LBoost)
WL=function(x,y,w) {
  model = lm(y~x-1,weights = w)
  pred = predict(model)
  loss = sum(log(1+exp(- as.vector(y) * pred)))
  return(as.numeric(loss))
}
rm(h_test)

# LBoost(n_iter = 500)
boost_iter = c(10,30,100,300,500)
rm(Final_LBoost)
Final_LBoost = function(boost_iters=boost_iter,data_train,data_test,n_cores=9)
{
  # my.cluster = makeCluster(n_cores)
  # registerDoParallel(my.cluster)
  # invisible(clusterEvalQ(my.cluster,{library(dplyr)
  #   library(stargazer)
  #   library(caret)
  #   library(pROC)
  #   library(ggplot2)
  #   library(gridExtra)
  #   library(doParallel)
  #   }))
  # clusterExport(my.cluster,c("LBoost","data_train","data_test","WL"),envir = .GlobalEnv)
  result = lapply(boost_iters, LBoost,data_train,data_test,n_cores)
  
 # stopCluster(my.cluster)
  output = list()
  output$loss_500= result[[5]]$loss.plot 
  output$roc_100 = result[[3]]$roc.plot
  train_miss = array()
  test_miss = array()
  for(i in 1:length(boost_iters))
  {
    train_miss[i] = result[[i]]$Train_miss
    test_miss[i] = result[[i]]$Test_miss
  }
  D=data.frame(Iteration=boost_iters,Miss_Train = train_miss,Miss_Test = test_miss)
  output$Result = D
  output$Misclassification.plot = ggplot(D,aes(x=as.vector(boost_iters)))+
    geom_line(aes(y=as.vector(train_miss),color="Train"))+
    geom_line(aes(y=as.vector(test_miss),color="Test"))+ 
    ylab('Misclassification Error')+ xlab('Number of Feature')
  return(output)
}
R=Final_LBoost(boost_iter,data_train,data_test,n_cores=9)
R$Result
R$loss_500
R$roc_100
R$Misclassification.plot
job({rmarkdown::render("HW_6.1.R", "pdf_document")})
