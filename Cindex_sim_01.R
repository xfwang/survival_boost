#generate train data
library(xgboost)
n <- 400; p <- 100
beta <- c(rep(1,10),rep(0,p-10))
x <- matrix(rnorm(n*p),n,p)
real.time <- -(log(runif(n)))/(10*exp(drop(x %*% beta)))
cens.time <- rexp(n,rate=1/10)
status <- ifelse(real.time <= cens.time,1,0)
obs.time <- ifelse(real.time <= cens.time,real.time,cens.time)
#convert to xgboost data structure
t=200
dtrain<-list(data=x[1:t,],label=obs.time[1:t])
attr(dtrain,'censor')<-status[1:t]
Dtrain<-xgb.DMatrix(dtrain$data,label=dtrain$label)
attr(Dtrain,"censor")<-attr(dtrain,"censor")
#Dtrain2<-xgb.DMatrix(dtrain$data[,1:10],label=dtrain$label)
#attr(Dtrain2,"censor")<-attr(dtrain,"censor")

#generate test data
#n2 <- 200; p2 <- 100
#beta2 <- c(rep(1,10),rep(0,p2-10))
#x2 <- matrix(rnorm(n2*p2),n2,p2)
#real.time2 <- -(log(runif(n2)))/(10*exp(drop(x2 %*% beta2)))
#cens.time2 <- rexp(n2,rate=1/10)
#status2 <- ifelse(real.time2 <= cens.time2,1,0)
#obs.time2 <- ifelse(real.time2 <= cens.time2,real.time2,cens.time2)
#convert to xgboost data structure
t=200
dtest<-list(data=x[(t+1):n,],label=obs.time[(t+1):n])
attr(dtest,'censor')<-status[(t+1):n]
Dtest<-xgb.DMatrix(dtest$data,label=dtest$label)
attr(Dtest,"censor")<-attr(dtest,"censor")
#Dtest2<-xgb.DMatrix(dtest$data[,1:10],label=dtest$label)
#attr(Dtest2,"censor")<-attr(dtest,"censor")

library(xgboost)
#define objective function and evaluation function
mylossobj2<-function(preds, dtrain) {
  alpha=2
  labels <- getinfo(dtrain, "label") #labels<-dtrain$label
  #print(labels)
  censor<-attr(dtrain,"censor")
  ord<-order(labels)
  ran=rank(labels)
  #print(ord)
  #foo<<-censor
  #compute the first gradient
  d=censor[ord]  #status
  etas=preds[ord] #linear predictor
  n=length(etas)
  temp1l=matrix(etas[1:(n-1)],nrow=n-1,ncol=n-1)-matrix(etas[2:n],nrow=n-1,ncol=n-1,byrow=1)
  #print(temp1l)
  temp2l=exp(alpha)^temp1l
  temp2l[lower.tri(temp2l)]=0
  temp3l=(1+temp2l)^2
  #print(temp3l)
  temp4l=-alpha*temp2l/temp3l
  temp4l[lower.tri(temp4l)]=0
  #print(temp4l)
  temp4r=-temp4l
  gl=c(apply(temp4l,1,sum),0)*d
  gr=c(0,-apply(temp4l,2,sum))
  grad=gl+gr
  grad=grad[ran]
  #print(grad)
  #derive hessian  
  # H1=outer(haz,rsk^2,'/')
  # H1=t(t(H1)*rsk)
  temp5l=-(alpha*temp2l*temp3l-temp2l*2*temp3l*alpha*temp2l)/(1+temp2l)^4
  temp5l[lower.tri(temp5l)]=0
  hl=c(apply(temp5l,1,sum),0)*d
  hr=c(0,apply(temp5l,2,sum))
  hess=hl+hr 
  hess=hess[ran]
  #hess=rep(0,length(grad))
  # Return the result as a list
  return(list(grad = grad, hess = hess))
}

evalerror2 <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label") #labels<-dtrain$label
  censor<-attr(dtrain,"censor") #not working!
  #foo<<-censor
  #compute the first gradient
  ord<-order(labels)
  d=censor[ord]  #status
  etas=preds[ord] #linear predictor
  haz<-as.numeric(exp(etas)) #w[i]
  rsk<-rev(cumsum(rev(haz)))
  err <- -sum(d*(etas-log(rsk)))
  return(list(metric = "deviance", value = err))
}
#fit1<-xgboost(data = Dtrain, nrounds=350,objective = mylossobj2,eval_metric = evalerror2)
#xgb.importance(model=fit1)
##parameter tuning
best_param = list()
best_seednumber = 1234
best_loss = Inf
best_loss_index = 0

for (iter in 1:500) {
  param <- list(objective = mylossobj2,
                eval_metric = evalerror2,
                #num_class = 12,
                max_depth = sample(6:13, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, 1), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1),
                colsample_bylevel=runif(1, .5, 1),
                lambda=runif(1,0,2),
                alpha=runif(1,0,2)
  )
  cv.nround = 500
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=Dtrain, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = F)
  
  min_loss = min(mdcv$evaluation_log[,'test_deviance_mean'])
  min_loss_index = which.min(as.numeric(unlist(mdcv$evaluation_log[,'test_deviance_mean'])))
  
  if (min_loss < best_loss) {
    best_loss = min_loss
    best_loss_index = min_loss_index
    best_seednumber = seed.number
    best_param = param
  }
  print(iter)
}

nround = best_loss_index
set.seed(best_seednumber)
best_param$objective=mylossobj2
md <- xgboost(data=Dtrain, params=best_param, nrounds=nround,nthread=6)
#md2 <- xgboost(data=Dtrain2, params=best_param, nrounds=nround, nthread=6)
xgb.importance(model=md)
#predict(md,Dtrain)
#order(predict(md,Dtrain2))
#rev(order(obs.time))

#concordance index for test data
library(survival)
survConcordance(Surv(obs.time[1:t], status[1:t]) ~predict(md,Dtrain))
survConcordance(Surv(obs.time[(t+1):n], status[(t+1):n]) ~predict(md,Dtest))


#compare with gbm cox
library(gbm)
library(survival)
gbm1 <- gbm(Surv(obs.time[1:t],status[1:t])~ .,       # formula
            data=as.data.frame(x[1:t,]),                 # dataset
            #weights=w,
            #var.monotone=c(0,0,0),     # -1: monotone decrease, +1: monotone increase, 0: no monotone restrictions
            distribution="coxph",
            n.trees=1000,              # number of trees
            shrinkage=0.001,           # shrinkage or learning rate, 0.001 to 0.1 usually work
            #interaction.depth=3,       # 1: additive model, 2: two-way interactions, etc
            bag.fraction = 0.5,        # subsampling fraction, 0.5 is probably best
            train.fraction = 0.8,      # fraction of data for training, first train.fraction*N used for training
            cv.folds = 5,              # do 5-fold cross-validation
            #n.minobsinnode = 10,       # minimum total weight needed in each node
            keep.data = TRUE,
            verbose = TRUE)           #  print progress
summary(gbm1)
#aa=predict(gbm1,data=as.data.frame(x2))
survConcordance(Surv(obs.time[(t+1):n], status[(t+1):n]) ~ predict(gbm1,data=as.data.frame(x[(t+1):n,])))

#compare with coxboost
library(CoxBoost)
cbfit <- CoxBoost(time=obs.time[1:t],status=status[1:t],x=x[1:t,])
summary(cbfit)
#survConcordance(Surv(obs.time[1:t], status[1:t]) ~ as.vector(predict(cbfit)))
survConcordance(Surv(obs.time[(t+1):n], status[(t+1):n]) ~ as.vector(predict(cbfit,newdata=x[(t+1):n,])))

