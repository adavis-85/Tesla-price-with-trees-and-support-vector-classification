install.packages("ISLR")
library(ISLR)
install.packages("tree")
library(tree)
install.packages("gbm")
library(gbm)
install.packages("randomForest")
library(randomForest)
library(e1071)

attach(TSLA)

direction=rep("No",length(Close))

for(i in 1:252)
{
  if(Close[i]<Close[i+1])
  {
    direction[i]="Yes"
  }
}

new.TSLA=data.frame(direction,TSLA[c(2,3,4,5,7)])

tree.TSLA=tree(direction~.,new.TSLA)
summary(tree.TSLA)
plot(tree.TSLA)
text(tree.TSLA,pretty=0)

set.seed(2)
train=sample(1:nrow(new.TSLA),200)
new.test=new.TSLA[-train,]
direction.test=direction[-train]
tree.TSLA=tree(direction~.,new.TSLA,subset=train)
TSLA.prediction=predict(tree.TSLA,new.test,type="class")
table(TSLA.prediction,direction.test)


set.seed(3)
cv.TSLA=cv.tree(tree.TSLA,FUN=prune.misclass)
cv.TSLA
prunedTSLA=prune.misclass(tree.TSLA,best=8)
plot(prunedTSLA)
text(prunedTSLA,pretty=0)

prunedTSLApred=predict(prunedTSLA,new.test,type="class")
table(prunedTSLApred,direction.test)


names(TSLA)

Direction=rep(0,length(Close))

for(i in 1:252)
{
  if(Close[i]<Close[i+1])
  {
    Direction[i]=1
  }
}

data1=data.frame(x=TSLA[,c(2,3,4,5,7)],y=as.factor(Direction))
svmTSLA=svm(y~.,data=data1,kernel="linear",cost=10,scale=FALSE)
summary(svmTSLA)

set.seed(1)
tuneTSLA1=tune(svm,y~.,data=data1,kernel="linear",ranges = list(cost=c(.001,.01,.1,1,5,10,100)))
summary(tuneTSLA1)

bestmodel1=tuneTSLA1$best.model
summary(bestmodel1)

set.seed(2)
test=sample(1:nrow(data1),100)
xtest=TSLA[test,]
ytest=Direction[test]



data2=data.frame(x=TSLA[test,c(2,3,4,5,7),],y=as.factor(ytest))
predict1=predict(bestmodel1,data2)
table(predict=predict1,truth=data2$y)


tuneTSLA2=tune(svm,y~.,data=data2,kernel="radial",gamma=c(.01,.1,.5,1,2,3,4),ranges=list(cost=c(.001,.01,.1,1,5,10,100)))
summary(tuneTSLA2)

bestmodel2=tuneTSLA2$best.model
summary(bestmodel2)

predict2=predict(bestmodel2,data2)
table(predict=predict2,data2$y)

tuneTSLA3=tune(svm,y~.,data=data2,kernel="polynomial",gamma=c(.01,.1,.5,1,2,3,4),ranges=list(cost=c(.001,.01,.1,1,5,10,100)))
summary(tuneTSLA3)

bestmodel3=tuneTSLA3$best.model
summary(bestmodel3)

predict3=predict(bestmodel3,data2)
table(predict=predict3,data2$y)

