##  boosting 
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library(xgboost)
library(ggplot2)
library(gridExtra)
library(e1071)
library(mxnet)
library(entropy)

### data preparation #####
data_dummy = read.csv('/Users/nick/Desktop/oneDrive/论文答辩/data and R code/data_dummy.csv')[,2:68]
data_dummy_by_year = data_dummy[,c(1:7,13:67)]
data_dummy_by_year$year = as.matrix(data_dummy[,8:12])%*%matrix(1:5,5,1)+2010
data_dummy_by_year_claimed = data_dummy_by_year[data_dummy_by_year$inc > 0, ]
data_list_claimed = list(data_dummy_by_year_claimed[data_dummy_by_year_claimed$year==2010,], 
                         data_dummy_by_year_claimed[data_dummy_by_year_claimed$year==2011,], 
                         data_dummy_by_year_claimed[data_dummy_by_year_claimed$year==2012,], 
                         data_dummy_by_year_claimed[data_dummy_by_year_claimed$year==2013,], 
                         data_dummy_by_year_claimed[data_dummy_by_year_claimed$year==2014,])

##  Mean and SD ###
mean_inc=NULL
sd_inc = NULL
for(i in 1:6) mean_inc[i] = mean(data_list_claimed[[i]]$inc)
for(i in 1:6) sd_inc[i] = sd(data_list_claimed[[i]]$inc)
mean_inc
sd_inc

ggdata = data.frame(mean = round(mean_inc, 0),
                    sd = round(sd_inc,0), 
                    year = 2010:2014)
ggdata = ggdata[ggdata$year!=2015,]
ggplot(ggdata,aes(x = year, y = sd))+
  geom_line(colour = 'red4', size = 2) +
  geom_text(aes(x = year, y = sd, label =sd),vjust = 4, colour = 'black',position = position_nudge(y=0),size = 3.5)+
  geom_point(shape = 22, size = 4,fill = 'pink') +
  geom_bar(stat="identity", alpha=0.75, fill = 'grey7',width = 0.5, data = ggdata,aes(x = year, y = mean))+
  geom_text(aes(x = year, y = mean, label =mean),vjust = -1, colour = 'black',position = position_dodge(0.9),size = 3.5)+
  theme_bw()+
  scale_x_continuous(breaks = 2010:2015)+
  ylab('SD&MEAN')

###  main train function 
trees = 5
boosting_regression = function(train_data, test_data){
  small_train_x = train_data[,c(1:6,8:59)]
  small_train_x[,c('I4','I5','I6','O1','V4','V6')] = (small_train_x[,c('I4','I5','I6','O1','V4','V6')] - 
                                                        matrix(colMeans(small_train_x[,c('I4','I5','I6','O1','V4','V6')]), 
                                                               nrow = dim(small_train_x[,c('I4','I5','I6','O1','V4','V6')])[1],
                                                               ncol = dim(small_train_x[,c('I4','I5','I6','O1','V4','V6')])[2],
                                                               byrow = T))/matrix(sqrt(diag(var(small_train_x[,c('I4','I5','I6','O1','V4','V6')]))),
                                                                                  nrow = dim(small_train_x[,c('I4','I5','I6','O1','V4','V6')])[1],
                                                                                  ncol = dim(small_train_x[,c('I4','I5','I6','O1','V4','V6')])[2],byrow = T)
  small_train_weight = train_data$I2[rowSums(small_train_x[,c('I4','I5','I6','O1','V4','V6')] > 3) == 0]
  small_train_y = train_data$inc[rowSums(small_train_x[,c('I4','I5','I6','O1','V4','V6')] > 3) == 0]
  small_train_x = small_train_x[rowSums(small_train_x[,c('I4','I5','I6','O1','V4','V6')] > 3) == 0, ]
  
  small_test_weight = test_data$I2
  small_test_x = test_data[small_test_weight == 1,c(1:6,8:59)]
  small_test_x[,c('I4','I5','I6','O1','V4','V6')] = (small_test_x[,c('I4','I5','I6','O1','V4','V6')] - 
                                                       matrix(colMeans(small_test_x[,c('I4','I5','I6','O1','V4','V6')]), 
                                                              nrow = dim(small_test_x[,c('I4','I5','I6','O1','V4','V6')])[1],
                                                              ncol = dim(small_test_x[,c('I4','I5','I6','O1','V4','V6')])[2],
                                                              byrow = T))/matrix(sqrt(diag(var(small_test_x[,c('I4','I5','I6','O1','V4','V6')]))),
                                                                                 nrow = dim(small_test_x[,c('I4','I5','I6','O1','V4','V6')])[1],
                                                                                 ncol = dim(small_test_x[,c('I4','I5','I6','O1','V4','V6')])[2],byrow = T)
  
  small_test_y =  test_data$inc[small_test_weight == 1]
  
  alpha = 0.7
  x = small_train_x
  y = small_train_y
  weight = small_train_weight
  test_set = small_test_x
  names(test_set) = names(small_train_x)
  test_label = small_test_y
  nround = 300
  
  output_gbt = test_label
  output_mlp = test_label
  output_svm = test_label
  output_glm = test_label
  output_gbt_01 = test_label
  output_glm_01 = test_label
  output_train = NULL
  nrows = length(y)
  for(tree in 1:trees){
    boostrap = sample(1:nrows, size = alpha*nrows, replace = T, prob = weight)
    x[boostrap[1],diag(var(x[boostrap,]))==0] = 1
    boostrap_list = list(x[boostrap,],log(y[boostrap]))
    params = list(max.depth = 500, eta = 0.3,
                  objective='reg:linear',eval_metric='rmse')
    train_data = xgb.DMatrix(as.matrix(boostrap_list[[1]]),label = boostrap_list[[2]])
    xgb_fit = xgb.train(params = params,
                        data = train_data,
                        nrounds = nround, verbose = 1, print.every.n = 10)
    pre_test = predict(xgb_fit, newdata = as.matrix(test_set))
    svm_fit = svm(x=boostrap_list[[1]], y=boostrap_list[[2]], cost = 0.1)
    pre_svm = predict(svm_fit, newdata = as.matrix(test_set))
    pre_train = predict(xgb_fit,newdata = as.matrix(small_train_x))
    output_train = cbind(output_train,pre_train)
    output_gbt = cbind(output_gbt,exp(pre_test))
    print(tree)
    output_svm = cbind(output_svm,pre_svm)
    
    mlp_fit = mx.mlp(as.matrix(boostrap_list[[1]]),boostrap_list[[2]], hidden_node = c(8, 6), out_node = 1, 
                     num.round=100, array.batch.size=200, learning.rate=0.04, momentum=0.7,
                     out_activation="rmse",activation = 'relu',dropout = 0.8,
                     eval.metric=mx.metric.rmse)
    
    pre_mlp = predict(mlp_fit, as.matrix(test_set))
    output_mlp = cbind(output_mlp, exp(as.vector(pre_mlp)))
    SVM_fit = svm(as.matrix(boostrap_list[[1]]),boostrap_list[[2]], cost = 0.02, 
                  kernel = "sigmoid", type = 'eps-regression')
    pre_svm = predict(SVM_fit, as.matrix(small_test_x))
    output_svm = cbind(output_svm, exp(as.vector(pre_svm)))
  }
  
  glm_data = data.frame(small_train_x, y = small_train_y)
  glm_data = rbind(glm_data, colMeans(glm_data) + 0.01)
  names(glm_data) = c(names(small_test_x),'y')
  glm_fit = glm(y~., data = glm_data, family=Gamma(link="log"), 
               # weight = c(small_train_weight,1), 
                offset = log(c(small_train_weight,1)),
                control = list(epsilon = 1e-12, maxit = 250, trace = FALSE))
  pre_glm = predict(glm_fit, newdata = test_set, type ="response")
  pre_glm[pre_glm > 200000] = 200000
  pre_glm_01 = predict(glm_fit, newdata = test_set, type ="response")
  output_glm = cbind(output_glm,pre_glm)
  
  return(list(output_gbt,output_glm, output_mlp, output_svm))
  # return(list(output_gbt,output_svm))
}

### trainin process
output_reg_list = NULL
for(i in 1:4){
  output_reg_list[[i]] = boosting_regression(data_list_claimed[[i]], data_list_claimed[[i+1]])
  print(i)
}


###  RMSE ####
rmse_gbt = NULL
rmse_glm = NULL
rmse_mlp = NULL
rmse_svm = NULL
rmse = function(yhat,y) sqrt(mean((y-yhat)^2,na.rm = T))

for(i in 1:4){
  rmse_gbt = c(rmse_gbt, rmse(rowMeans(output_reg_list[[i]][[1]][,2:(trees+1)]), output_reg_list[[i]][[1]][,1]))
  rmse_glm = c(rmse_glm, rmse(output_reg_list[[i]][[2]][,2], output_reg_list[[i]][[2]][,1]))
  rmse_mlp = c(rmse_mlp, rmse(rowMeans(output_reg_list[[i]][[3]][,2:(trees+1)]), output_reg_list[[i]][[3]][,1]))
  rmse_svm = c(rmse_svm, rmse(rowMeans(output_reg_list[[i]][[4]][,2:(trees+1)]), output_reg_list[[i]][[4]][,1]))
}

rmse_gbt
rmse_glm
rmse_mlp
rmse_svm
ggdata = data.frame(rmse = c(rmse_gbt,rmse_glm, rmse_mlp, rmse_svm), 
                    model = c(rep('gbt',4),rep('glm',4), rep('mlp',4), rep('svm',4)), 
                    year = c(2011:2014,2011:2014,2011:2014,2011:2014))
ggplot(ggdata,aes(x = year, y = rmse, fill = model)) + 
  geom_bar(stat = "identity",position="dodge")+
  scale_fill_grey(start = 0.4, end = 0)+
  theme_bw()+
  geom_text(aes(x = year, y = rmse, label = round(rmse,0)) ,vjust = 1.5, colour = 'white',position = position_dodge(0.9),size = 3)
#  coord_cartesian(ylim=c(10000,16000))
#  xlab('logistic regression')+ylab('ensemble learning')
  


##########################################
#####  glm * mlp

p = NULL
for(i in 1:4){
  ggdata = data.frame(x = log(output_reg_list[[i]][[2]][,2]), y = log(rowMeans(output_reg_list[[i]][[3]][,2:(trees+1)])))
  ggdata = ggdata[rbinom(dim(ggdata)[1], 1, 0.1) == 1,]
  ggdata2 = data.frame(x = quantile(log(output_reg_list[[i]][[2]][,2]), seq(0.005, 0.995, by = 0.01)),
                       y = quantile(log(rowMeans(output_reg_list[[i]][[3]][,2:(trees+1)])), seq(0.005, 0.995, by = 0.01)))
  
  p[[i]]=
    ggplot(ggdata, aes(x=x, y=y)) + 
    geom_point(alpha=0.4) +
    ylim(c(quantile(ggdata$y,0.005),quantile(ggdata$y,0.995))) + 
    xlim(c(quantile(ggdata$x,0.005), quantile(ggdata$x, 0.995))) +
    geom_line(data = ggdata2, aes(x = x, y = y), colour = 'red', size = 1.5)+
    theme_bw() +
    xlab('glm prediction') +
    ylab('mlp prediction') + 
    ggtitle(paste('Year', i+2010))
}
grid.arrange(p[[1]],p[[2]],p[[3]],p[[4]],ncol = 4)

##########################################
##### glm * gbt
p = NULL
for(i in 1:4){
  ggdata = data.frame(x = log(output_reg_list[[i]][[2]][,2]), y = log(rowMeans(output_reg_list[[i]][[1]][,2:(trees+1)])))
  ggdata = ggdata[rbinom(dim(ggdata)[1], 1, 0.1) == 1,]
  ggdata2 = data.frame(x = quantile(log(output_reg_list[[i]][[2]][,2]), seq(0.005, 0.995, by = 0.01)),
                       y = quantile(log(rowMeans(output_reg_list[[i]][[1]][,2:(trees+1)])), seq(0.005, 0.995, by = 0.01)))
  
  p[[i]]=
    ggplot(ggdata, aes(x=x, y=y)) + 
    geom_point(alpha=0.4) +
    ylim(c(quantile(ggdata$y,0.005),quantile(ggdata$y,0.995))) + 
    xlim(c(quantile(ggdata$x,0.005), quantile(ggdata$x, 0.995))) +
    geom_line(data = ggdata2, aes(x = x, y = y), colour = 'red', size = 1.5)+
    theme_bw() +
    xlab('glm prediction') +
    ylab('gbt prediction') + 
    ggtitle(paste('Year', i+2010))
}
grid.arrange(p[[1]],p[[2]],p[[3]],p[[4]],ncol = 4)

##########################################
##### glm * svm
p = NULL
for(i in 1:4){
  ggdata = data.frame(x = log(output_reg_list[[i]][[2]][,2]), y = log(rowMeans(output_reg_list[[i]][[4]][,2:(trees+1)])))
  ggdata = ggdata[rbinom(dim(ggdata)[1], 1, 0.1) == 1,]
  ggdata2 = data.frame(x = quantile(log(output_reg_list[[i]][[2]][,2]), seq(0.005, 0.995, by = 0.01)),
                       y = quantile(log(rowMeans(output_reg_list[[i]][[4]][,2:(trees+1)])), seq(0.005, 0.995, by = 0.01)))
  
  p[[i]]=
    ggplot(ggdata, aes(x=x, y=y)) + 
    geom_point(alpha=0.4) +
    ylim(c(quantile(ggdata$y,0.005),quantile(ggdata$y,0.995))) + 
    xlim(c(quantile(ggdata$x,0.005), quantile(ggdata$x, 0.995))) +
    geom_line(data = ggdata2, aes(x = x, y = y), colour = 'red', size = 1.5)+
    theme_bw() +
    xlab('glm prediction') +
    ylab('svm prediction') + 
    ggtitle(paste('Year', i+2010))
}
grid.arrange(p[[1]],p[[2]],p[[3]],p[[4]], ncol = 4)


#### KL divergence ############
kl_svm = NULL
kl_gbt = NULL
kl_glm = NULL
kl_mlp = NULL

for(i in 1:4){
  kl_gbt[i] = KL.empirical(rowMeans(output_reg_list[[i]][[1]][,2:(trees+1)]), output_reg_list[[i]][[1]][,1])
  kl_glm[i] = KL.empirical(output_reg_list[[i]][[2]][,2], output_reg_list[[i]][[2]][,1])
  kl_mlp[i] = KL.empirical(rowMeans(output_reg_list[[i]][[3]][,2:(trees+1)]), output_reg_list[[i]][[3]][,1])
  kl_svm[i] = KL.empirical(rowMeans(output_reg_list[[i]][[4]][,2:(trees+1)]), output_reg_list[[i]][[4]][,1])
}

kl_svm 
kl_gbt
kl_glm 
kl_mlp

ggdata = data.frame(kl.divergence = c(kl_gbt, kl_glm, kl_mlp, kl_svm), 
                    model = c(rep('gbt',4),rep('glm',4), rep('mlp',4), rep('svm',4)), 
                    year = c(2011:2014,2011:2014,2011:2014,2011:2014))
ggplot(ggdata,aes(x = year, y = kl.divergence, fill = model)) + 
  geom_bar(stat = "identity",position="dodge")+
  scale_fill_grey(start = 0.4, end = 0)+
  theme_bw()+
  geom_text(aes(x = year, y = kl.divergence, label = round(kl.divergence,3)) ,
            vjust = 1.5, colour = 'white',position = position_dodge(0.9),size = 3)+
  coord_cartesian(ylim=c(0.8,1.4))+
  ylab('KL divergence')


# write.csv(output_reg_list[[1]], '/Users/nick/Desktop/output_sev_1.csv')
# write.csv(output_reg_list[[2]], '/Users/nick/Desktop/output_sev_2.csv')
# write.csv(output_reg_list[[3]], '/Users/nick/Desktop/output_sev_3.csv')
# write.csv(output_reg_list[[4]], '/Users/nick/Desktop/output_sev_4.csv')




