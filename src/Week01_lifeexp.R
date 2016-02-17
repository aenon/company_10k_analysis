# For R and you Joel


setwd("/Users/helenliu/Documents/Berkeley/242/Material for students /Datasets/StateData.csv")
library(ggplot2)
library(dplyr)
library(reshape2)


state=read.csv("/Users/helenliu/Documents/Berkeley/242/Material for students /Datasets/StateData.csv")
str(state)
summary(state)

#Check for missing values
colSums(is.na(state))



#Correlation Plot
state1=state
qplot(x=Var1,y=Var2, data=melt(cor(state1)), fill=value, geom="tile") + geom_text(aes(Var2,Var1,label=round(value,digits =2)),color="white",size=5)
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
pairs(state1, lower.panel=panel.smooth)
pairs(state1, lower.panel = panel.smooth, upper.panel = panel.cor)

#1
sp1=ggplot(state, aes(x=Longitude, y=Latitude)) + geom_point(color="pink",size=2)
plot(state$Longitude,state$Latitude)
print(sp1)

#2
state %>% group_by(Region) %>% summarise(mean(HighSchoolGrad))
tapply(state$HighSchoolGrad, state$Region, mean)

#3
bp1=boxplot(state$Murder~state$Region)
bp1.stats=bp1$stats
colnames(bp1.stats)=bp1$names
rownames(bp1.stats)=c("Minumum", "Lower Quatile", "Median", "Upper Quartile", "Maximum")
bp1.stats=t(bp1.stats)
bp1.stats=as.data.frame(bp1.stats)
bp1.stats$Range=bp1.stats$Maximum-bp1.stats$Minumum
bp1.stats

bp1=ggplot(state, aes(x=Region,y=Murder,fill=Region)) +geom_boxplot()
print(bp1)

#4
model.1 = lm(LifeExp ~ Population + Income + Illiteracy + Murder + HighSchoolGrad + Frost + Area + Region + Longitude + Latitude, data=state)
summary(model.1)$r.squared
summary(model.1)
sp2=ggplot(state,aes(x=Income,y=LifeExp))+geom_point(color="red",size=2)+ geom_smooth(method='lm',formula=y~x)
print(sp2)

model.2=lm(LifeExp ~ Population + Murder + HighSchoolGrad + Frost, data=state)
summary(model.2)
summary(model.2)$r.squared

model.3=lm(LifeExp ~ Population + Murder + HighSchoolGrad + Region + Longitude + Latitude, data=state)
summary(model.3)
summary(model.3)$r.squared

step(model.1,direction="both")
#Lower the AIC, better is the model

prediction=predict(model.3,newdata=state)
state$Prediction=prediction
state$State=state.abb
View(state)

subset(state,select=c(State,LifeExp,Prediction),state$LifeExp==max(state$LifeExp))
subset(state,select=c(State,LifeExp,Prediction),state$LifeExp==min(state$LifeExp))
subset(state,select=c(State,LifeExp,Prediction),state$Prediction==max(state$Prediction))
subset(state,select=c(State,LifeExp,Prediction),state$Prediction==min(state$Prediction))


