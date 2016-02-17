# Import libraries
library(ggplot2)
library(dplyr)
library(reshape2)

# Read data
climate=read.csv("ClimateChange.csv")

# Understand structure of data and summarize it
str(climate)
summary(climate)

# Check for missing values
colSums(is.na(climate))

#Create a training and test set
trainingset <- climate[1:284,]
testset <- climate[285:308,]

# Fit a basic model
model1 <- lm(Temp ~ MEI + CO2 + CH4 + N2O + CFC.11 + CFC.12 + TSI + Aerosols, data = trainingset)
summary(model1)

# Joel's code to obtain correlations between independent variables
trainingset1=trainingset
trainingset1$Year=NULL
trainingset1$Month=NULL
qplot(x=Var1,y=Var2, data=melt(cor(trainingset1)), fill=value, geom="tile") + geom_text(aes(Var2,Var1,label=round(value,digits =2)),color="white",size=5)
panel.cor <- function(x, y, digits=2, prefix="", cex.cor, ...)
{
    usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
    r <- abs(cor(x, y))
    txt <- format(c(r, 0.123456789), digits=digits)[1]
      txt <- paste(prefix, txt, sep="")
      if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
        text(0.5, 0.5, txt, cex = cex.cor * r)
}
pairs(trainingset1, lower.panel=panel.smooth)
pairs(trainingset1, lower.panel = panel.smooth, upper.panel = panel.cor)
cor(trainingset1)
pairs(trainingset, lower.panel = panel.smooth, upper.panel = panel.cor)

# Fit improved model
model2 <- lm(Temp ~ MEI + TSI + Aerosols + N2O, data=trainingset)
summary(model2)

# Use new model to make predictions for test set
prediction=predict(model2,newdata=testset)
summary(prediction)
testset$Model2.Prediction=prediction
View(testset)

# Compute R2 for prediction
sse=sum((testset$Model2.Prediction-testset$Temp)^2)
sst=sum((mean(trainingset$Temp)-testset$Temp)^2)
rsquare=1-(sse/sst)
rsquare

#Plots - Model 2
testsetplot2007=melt(testset[1:12,c("Year","Month","Temp", "Model2.Prediction")], id.vars=c(1,2))
testsetplot2008=melt(testset[12:24,c("Year","Month","Temp", "Model2.Prediction")], id.vars=c(1,2))
ggplot(testsetplot2007,aes(x=as.factor(Month),y=value)) + geom_bar(aes(fill=variable),stat="identity", position = "dodge") + xlab("Months 2007")
ggplot(testsetplot2008,aes(x=as.factor(Month),y=value)) + geom_bar(aes(fill=variable),stat="identity", position = "dodge") + xlab("Months 2008")

#Best Model
step(lm(Temp~.,data=trainingset1),direction="both")
model3=lm(formula = Temp ~ MEI + CO2 + N2O + CFC.11 + CFC.12 + TSI + Aerosols, data = trainingset1)
prediction=predict(model3,newdata=testset)
testset$Model3.Prediction=prediction

sse=sum((testset$Model3.Prediction-testset$Temp)^2)
sst=sum((mean(trainingset$Temp)-testset$Temp)^2)
rsquare=1-(sse/sst)
rsquare

#Plots - Model 3
testsetplot2007=melt(testset[1:12,c("Year","Month","Temp", "Model3.Prediction")], id.vars=c(1,2))
testsetplot2008=melt(testset[12:24,c("Year","Month","Temp", "Model3.Prediction")], id.vars=c(1,2))
ggplot(testsetplot2007,aes(x=as.factor(Month),y=value)) + geom_bar(aes(fill=variable),stat="identity", position = "dodge") + xlab("Months 2007")
ggplot(testsetplot2008,aes(x=as.factor(Month),y=value)) + geom_bar(aes(fill=variable),stat="identity", position = "dodge") + xlab("Months 2008")

