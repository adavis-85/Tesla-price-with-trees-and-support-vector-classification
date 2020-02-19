# Tesla price with trees and support vector classification

Using trees a prediction will be made on the direction of Tesla stock.  First 
a variable will need to be made again.  To be able to classify using the tree mode
there will need to be a "Yes" or "No" class for up or down.  
```
direction=rep("No",length(Close))

for(i in 1:252)
{
  if(Close[i]<Close[i+1])
  {
    direction[i]="Yes"
  }
}
```
For this model the opening,high,low,close and volume variables will be used.
Because the next time periods direction is going to be predicted all of the 
variables will be used.  A new data frame will be created and the model will be graphed

```
new.TSLA=data.frame(direction,TSLA[c(2,3,4,5,7)])

tree.TSLA=tree(direction~.,new.TSLA)
summary(tree.TSLA)
plot(tree.TSLA)
text(tree.TSLA,pretty=0)

Classification tree:
tree(formula = direction ~ ., data = new.TSLA)
Variables actually used in tree construction:
[1] "Open"   "Volume"
Number of terminal nodes:  5 
Residual mean deviance:  1.314 = 325.8 / 248 
Misclassification error rate: 0.415 = 105 / 253 

```
![image](https://user-images.githubusercontent.com/58529391/74867063-dc5abf80-5308-11ea-95a3-b5cdc6e237fd.png)

Next a prediction will be made using sampling through the data.
```
set.seed(2)
train=sample(1:nrow(new.TSLA),200)
new.test=new.TSLA[-train,]
direction.test=direction[-train]
tree.TSLA=tree(direction~.,new.TSLA,subset=train)
TSLA.prediction=predict(tree.TSLA,new.test,type="class")
table(TSLA.prediction,direction.test)

               direction.test
TSLA.prediction No Yes
            No  11  25
            Yes  8   9
```
The first prediction for the tree is 56%.  Now the best tree size will be chosen
through cross-validation and pruned to that size.
```
set.seed(3)
cv.TSLA=cv.tree(tree.TSLA,FUN=prune.misclass)
cv.TSLA

$size
[1] 14 10  8  7  3  2  1

$dev
[1]  93  94  92  95 103  97 117

$k
[1]  -Inf  0.00  1.00  2.00  4.25  5.00 17.00

$method
[1] "misclass"

attr(,"class")
[1] "prune"         "tree.sequence"

prunedTSLA=prune.misclass(tree.TSLA,best=8)
plot(prunedTSLA)
text(prunedTSLA,pretty=0)

```
![image](https://user-images.githubusercontent.com/58529391/74867036-d1a02a80-5308-11ea-95a7-2cd9c22b4435.png)

Also the pruned model will be used to make a prediction.
```
prunedTSLApred=predict(prunedTSLA,new.test,type="class")
table(prunedTSLApred,direction.test)

prunedTSLApred No Yes
           No  11  25
           Yes  8   9
```
The accuracy is 37% a definite move in the wrong direction.  The model performed
better without being pruned .  

Testing will also be done using support vector machines for classification.
Another variable will be needed to show the direction up or down using a 
binary classification.  The resulting data frame will be used to train a 
model with a linear separation.  
```
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

```
Now the svm model will be tuned using a range of costs to choose the best one
```
set.seed(1)
tuneTSLA1=tune(svm,y~.,data=data1,kernel="linear",ranges = list(cost=c(.001,.01,.1,1,5,10,100)))
summary(tuneTSLA1)

Parameter tuning of ‘svm’:

- sampling method: 10-fold cross validation 

- best parameters:
 cost
    5

- best performance: 0.486 

- Detailed performance results:
   cost     error dispersion
1 1e-03 0.5641538 0.12317955
2 1e-02 0.5526154 0.11060006
3 1e-01 0.5018462 0.10518898
4 1e+00 0.5016923 0.11451972
5 5e+00 0.4860000 0.11128193
6 1e+01 0.4900000 0.10609375
7 1e+02 0.5329231 0.09696024
```
A cost of 5 has the least errror for the model.  Now it will be used to choose the 
best model for two different kernels, radial and polynomial.  The kernel is the way
that the classes are divided.  Linear is with a linear boundary and margin, radial 
us a bound between the classes and a polynomial meanders between the classes.
```
bestmodel1=tuneTSLA1$best.model

set.seed(2)
test=sample(1:nrow(data1),100)
xtest=TSLA[test,]
ytest=Direction[test]



data2=data.frame(x=TSLA[test,c(2,3,4,5,7),],y=as.factor(ytest))
predict1=predict(bestmodel1,data2)
table(predict=predict1,truth=data2$y)

      truth
predict  0  1
      0 11  5
      1 42 42
     
```
The accuracy using the best model with a linear kernel is 53%.  Now to test with the 
radial kernel and also a range of gamma and costs.
```
tuneTSLA2=tune(svm,y~.,data=data2,kernel="radial",gamma=c(.01,.1,.5,1,2,3,4),ranges=list(cost=c(.001,.01,.1,1,5,10,100)))
bestmodel2=tuneTSLA2$best.model
predict2=predict(bestmodel2,data2)
table(predict=predict2,data2$y)

predict  0  1
      0 47 27
      1  6 20
 ```
 The accuracy rate for the radial model is 67%.  The rate is an improvement but
 only for the direction going down being correctly predicted.  The prediction rate for 
 the direction going up has gone down more than 50%.  
 Now for a polynomial kernel.
 ```
 tuneTSLA3=tune(svm,y~.,data=data2,kernel="polynomial",gamma=c(.01,.1,.5,1,2,3,4),ranges=list(cost=c(.001,.01,.1,1,5,10,100)))
 bestmodel3=tuneTSLA3$best.model
 predict3=predict(bestmodel3,data2)
 table(predict=predict3,data2$y)
 
 predict  0  1
      0 53 44
      1  0  3
 ```
 The accuracy for the polynomial kernel is 56%.  The accuracy for a zero being predicted correctly has gone up 6% and the 
 accuracy for a one being predicted correctly has gone down 17%.  The best performing model was the svm with a radial
 kernel.  It performed better than a tree model though the performance for certain choices is not ideal.
