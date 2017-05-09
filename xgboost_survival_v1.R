
library(xgboost)
#define objective function and evaluation function
mylossobj2<-function(preds, dtrain) {
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
  haz<-as.numeric(exp(etas)) #w[i]
  #print(haz)
  rsk<-rev(cumsum(rev(haz))) #W[i]
  P<-outer (haz,rsk,'/')
  P[upper.tri(P)] <- 0
  grad<- -(d-P%*%d)
  grad=grad[ran]
  #derive hessian  
  # H1=outer(haz,rsk^2,'/')
  # H1=t(t(H1)*rsk)
  H1=P
  H2=outer(haz^2,rsk^2,'/')
  H=H1-H2
  H[upper.tri(H)]=0
  hess=H%*%d  
  hess=hess[ran]
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
