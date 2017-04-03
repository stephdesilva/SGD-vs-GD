#####################################################################
# Title:Stochastic Gradient Descent vs Gradient Descent: Seeing the difference
# Author: Steph de Silva.
# Email: steph@rex-analytics.com
# Date created: 18/03/17
# Date last altered: 18/03/17
# Attributions and acknowledgment of derivation:
# This script is due to information, tips and advice given in:
# - Generating random numbers in R: http://www.cookbook-r.com/Numbers/Generating_random_numbers/
# Purpose: This script generates a data generating process and fits a model
# with both gradient descent and stochastic gradient descent
#########################################################################
# Data Used: Generated
# Source: the computer
# Specifically: NA
# Translation by:NA
# Date Accessed: NA
# Gutenberg Number: NA
#########################################################################
# Script Outline:
# 1. Load Libraries
# 2. Generate DGP
# 3. Fit model with Gradient Descent
# 4. Fit model with SGD
# 5. Plot differences
#########################################################################
# 1. Load libraries, define control parameters
#########################################################################

rm(list=ls(all=TRUE)) # clear the workspace of anything else
set.seed(1234) # we may want to do this again some day. 
library(ggplot2)
setwd("~/Documents/Rex Analytics/Blog/ML theory")
n = 100
mean.x1 = 1
sd.x1 = 1
beta.0 =0
beta.1 = 1
k = 1
options = "t"
joption = 1

kstar = 2 # Number of x's to estimate on +1

tolerance = 0.0000001 # the delta that defines convergence in the algorithms
stepsize = 0.0001    # stepsize in the descent algorithms
max.iterations = 10000 # max iterations allowed in the descent algorithms
batch.size = 10 # batch size for batched stochastic gradient descent

#########################################################################
# 2. Generate DGP
#########################################################################

# Here, our true DGP is y(i)= beta.0 +beta.1*x.1(i)+epsilon(i)
# epsilon(i) is iid with a mean of zero and variance constant
# there are several options for epsilon(i) we will explore
# (1) standard normal (Plain vanilla)
# (2) t(4) - all required moments for generalised estimation
# exist, but the tails are VERY fat. The distribution is symmmetric
# like normal. Can we break either method?
# (3) centred and standardised chi squared (2). It's skewed, fat
# tailed and very, very difficult to estimate with.

# To begin with, we will assume that our true beta.0 =0 and beta.1=1
# this means the DGP is actually y(i)= x.1(i)+epsilon(i)
# we can alter this later.

# Define a function that will allow us to generate the data
# Generate x.1 as normal (1, 1), e.g. mean of 1, variance/SD of 1
# Some notation for our function:
# n = length of our series
# mean.x1 = mean of x.1
# sd.x1 = SD of x.1 these two parameters allow us to control signal/noise ratio
# beta.0 intercept of the true DGP
# beta.1 coefficient of x.1 in the true DGP
# k = number of independent variables to generate. Only x1 will be
# part of the true DGP of y, so we will have k-1 additional variables
# which may be spuriously estimated
# options = normal, t, chi2 - what is the DGP of errors?
# joption = degrees of freedom for error process defined by options
# superfluous for normal.
# kstar = the number of x variables used to estimate y +1 (for the constant). This may be different
# to the "real" k.

DGP.process = function (n, mean.x1, sd.x1, beta.0, beta.1, k, options, joption){
    # Generate an error term which is zero mean
    if (options=="normal"){
    epsilon = rnorm(n, mean=0, sd=1)
  } else if (options == "t"){
    epsilon = rt(n, joption, ncp=0)
    }else {
    epsilon1 = rchisq(n, joption, ncp=0) # OK some backgroud. This does not have mean zero
    epsilon = (epsilon1 - joption) /sqrt(2*joption)  # the non central chi squared dist has moments
                                          # mu1 = joptions + ncp (non centrality parameter)
                                          # mu 2= (joptions +ncp)^2 + 2(joptions +2*ncp)
                                          # As ncp is zero here we have mean = joptions
                                          # and sd = sqrt(mu2-mu1^2)
                                          #        = sqrt(joptions ^2 + 2*joptions - joptions^2)
                                          #        = sqrt(2*joptions)
                                          # to standardise: epsilon = (epsilon1- mu1)/sd
                                          # More detail? C.f. https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#Properties
    }
  # Generate the k independent variables x
  x={}
  for (i in 1:k){
    xi = rnorm(n, mean=mean.x1, sd=sd.x1)
    x = cbind (x, xi)
  }
  
  # Finally, generate the dependent variable y
  
  y = beta.0 +beta.1*x[,1] + epsilon
  
  # Combine into a single matrix of data, first column is y, second and subsequent is x
  y = cbind (y, x)
  return (y)
}

y = DGP.process (n, mean.x1, sd.x1, beta.0, beta.1, k, options, joption)

hist(y[,1])
hist(y[,2])

#########################################################################
# 3. Estimate y in a regression using gradient descent
#########################################################################
# This happens in two parts:
# a. define a gradient function
# b. Define a function for gradient descent
# On the theory side, check out: http://www.le.ac.uk/users/dsgp1/COURSES/MATHSTAT/13mlreg.pdf
# Note that we are oversimplifying here. We are not estimating sigma
# this means we don't have an understanding of how precise our estimates are
# not a good thing in machine learning or classic maximum likelihood

# So the log likelihood is proprtional to -1/2sigma^2 *(y - xbeta)'(y-xbeta)
# the gradient wrt to beta is 1/sigma^2(y-xbeta)'x
# note here x is the full matrix of kstar variables we are estimating with 
# and a column of ones for the intercept. Let's create that first:
# let's also just make y the dependent variable we're estimating
x = y[,2:kstar]
x = cbind (matrix(1, nrow=n, ncol=1), x)
y = y[,1]

# Function for the gradient
# Note we are just assuming sigma^2 is set to 1. This is 
# an oversimplification.
## working these out gave me some trouble but this post
# here: https://www.r-bloggers.com/regression-via-gradient-descent-in-r/
# was super helpful

gradient = function (y, x, beta){
  y = as.matrix(y)
  x = as.matrix(x)
  beta = as.matrix(beta)
  grad <- (1/nrow(y))* (t(x) %*% ((x %*% beta) - y))
  return (grad)
}

## Function for gradient descent
## here I'm using the euclidean distance between updates of estimated beta
## to decide if tolerance has been reached
## this isn't the only way.

gradient.descent = function (x, y, initial.beta, stepsize, tolerance, max.iterations){
  beta.descent = matrix(NA, nrow=max.iterations, ncol=kstar)
  cost.descent = matrix(NA, nrow=max.iterations, ncol=1)
  beta = initial.beta
  for (i in 1:max.iterations){
    new.beta = beta - stepsize*gradient(y, x, beta)
    cost.descent[i,1] <- t(x %*% beta - y)%*%(x %*% beta - y)/(2*n)
    beta = new.beta
    beta.descent[i,] = new.beta
    if (i>1){
      tolerance.now =abs(cost.descent[i,1]-cost.descent[i-1,1])
      if (tolerance.now - tolerance <= 0){
        beta.descent = cbind(beta.descent, cost.descent)
        break
      }
    }
  }
  beta.descent = cbind(beta.descent, cost.descent)
  return (beta.descent)
}

# With gradient descent, you need somewhere to start.
# initialise in this case randomly 
initial.beta = rnorm(kstar, mean = 0, sd=1)

gd.estimated.beta = gradient.descent (x, y,initial.beta, stepsize, tolerance, max.iterations )

plot (gd.estimated.beta[,2])
plot(gd.estimated.beta[,1])
plot(gd.estimated.beta[,3])

#########################################################################
# 3. Estimate y in a regression using stochastic gradient descent
#########################################################################
## We use a different gradient function to defined previously
# I'll use the same initialisation of beta we had previously
gradient.SGD = function (y, x, beta, k){
  y = as.matrix(y)
  x = as.matrix(x)
  beta = as.matrix(beta)
  grad <- (t(x[,k]) %*% ((x %*% beta) - y))
  return (grad)
}

stochastic.gradient.descent = function (x, y, initial.beta, stepsize, tolerance, max.iterations, batch.size){
  beta.descent = matrix(NA, nrow=max.iterations, ncol=kstar)
  cost.descent = matrix(NA, nrow=max.iterations, ncol=1)
  beta = initial.beta
  # didn't shuffle because this is all random, but should in practice.
  for (i in 1: max.iterations){
    for (j in 1:kstar){
      beta[j] = beta[j] - stepsize*gradient.SGD(y, x, beta, j)
    }
    cost.descent[i,1] <- t(x %*% beta - y)%*%(x %*% beta - y)/(2)
    beta.descent[i,] = beta
    if (i>1){
      tolerance.now =abs(cost.descent[i,1]-cost.descent[i-1,1])
      if (tolerance.now - tolerance <= 0){
        beta.descent = cbind(beta.descent, cost.descent)
        break
      }
    }
  }
  beta.descent = cbind(beta.descent, cost.descent)
  return (beta.descent)
}

sgd.estimated.beta = stochastic.gradient.descent (x, y,initial.beta, stepsize, tolerance, max.iterations )

plot (sgd.estimated.beta[,2])
plot(sgd.estimated.beta[,1])
plot(sgd.estimated.beta[,3])

#########################################################################
# 4. Need to manipulate output from each simulation to create facet plots
#########################################################################
# let's also compare the OLS estimates

OLS.output <- lm(y~x[,2:kstar])

beta1.output <- cbind(sgd.estimated.beta[,2], gd.estimated.beta[,2])
beta1.output <- as.data.frame(beta1.output)
beta1.output$Panel <- "Beta 1"
beta1.output$Iteration <- seq(from = 1, to =  max.iterations, by =1)
colnames (beta1.output) <- c("SGD", "GD", "Panel", "Iteration")

beta0.output<-cbind(sgd.estimated.beta[,1], gd.estimated.beta[,1])
beta0.output <- as.data.frame(beta0.output)
beta0.output$Panel <- "Beta 0"
beta0.output$Iteration <- seq(from = 1, to =  max.iterations, by =1)
colnames(beta0.output) <- c("SGD", "GD", "Panel", "Iteration")


caption.data <- paste("Generating process: ", options, "distribution" )
if (options!="normal"){
  caption.data <- paste (caption.data, " DF=", joption)
}
chart.data <- as.data.frame(rbind(beta1.output, beta0.output))
col_vec<-c("slategray3","plum4", "steelblue4")
p_ind<- ggplot(chart.data)+
  labs(x="Iteration", y="Estimated Value", caption=caption.data)+
  facet_grid(Panel~., scale="free")+
  geom_line(aes(Iteration, SGD, colour="SGD"), linetype=1)+
  geom_line(aes(Iteration, GD, colour="GD"), linetype=1)+
  theme(plot.margin = unit(c(1,1,1,1), "lines"))+
  theme_light()+
  scale_colour_manual(name="Index",values=col_vec)+
  theme(legend.position="bottom")+
  ggtitle("Stochastic Gradient Descent vs Gradient Descent")
print(p_ind)