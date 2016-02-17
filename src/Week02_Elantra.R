# Import required libraries
library(plyr)
library(ggplot2)

# Read data
elantra=read.csv("Elantra.csv")

# Understand data struture and summarize it
str(elantra)
summary(elantra)

# Check for missing values
colSums(is.na(elantra))

# Sort data in ascending order according to Year
elantra1 <- arrange(elantra,Year)

# Divide data into training set and test set
trainingset <- elantra1[1:36,]
summary(trainingset)
testset <- elantra1[37:50,]

#Models
model1 <- lm(ElantraSales ~ Unemployment + Queries + CPI.Energy + CPI.All, data = trainingset)
summary(model1)
model2 <- lm(ElantraSales ~ Unemployment + Queries + CPI.Energy + CPI.All + Month, data = trainingset)
summary(model2)

#Converting Month to factors
trainingset$Month=as.factor(trainingset$Month)
testset$Month=as.factor(testset$Month)

#New Model
model3 <- lm(ElantraSales ~ Unemployment + Queries + CPI.Energy + CPI.All + Month, data = trainingset)
summary(model3)

#Predictions
predictions=predict(model3,newdata=testset)
testset$Model3.Predictions=predictions
sse=sum((testset$Model3.Prediction-testset$ElantraSales)^2)
sst=sum((mean(trainingset$ElantraSales)-testset$ElantraSales)^2)
rsquare=1-(sse/sst)
rsquare

#Plots
testset$Year.Month=paste(testset$Year,testset$Month)
testset$Year.Month=factor(testset$Year.Month, levels=testset$Year.Month)
testsetplot=melt(testset[,c("Year.Month","ElantraSales","Model3.Predictions")],id.vars = 1)
ggplot(testsetplot,aes(x=Year.Month,y=value)) + geom_bar(aes(fill=variable),position = "dodge", stat="identity") + ylab("Sales")


