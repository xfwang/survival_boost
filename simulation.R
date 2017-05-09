library(xgboost)
n <- 200; p <- 100
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
Dtrain2<-xgb.DMatrix(dtrain$data[,1:10],label=dtrain$label)
attr(Dtrain2,"censor")<-attr(dtrain,"censor")


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
                #colsample_bylevel=runif(1, .5, 1),
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
md <- xgboost(data=Dtrain, params=best_param, nrounds=nround, nthread=6)
#md2 <- xgboost(data=Dtrain2, params=best_param, nrounds=nround, nthread=6)
xgb.importance(model=md)

