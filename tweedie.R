### preliminary
### install mxnet
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library(xgboost)
library(ggplot2)
library(gridExtra)
library(e1071)
library(mxnet)

data_dummy = read.csv('C:\\Users\\TIANBO001\\OneDrive\\master thesis\\data and R code\\data_dummy\'.csv')[,2:68]
data_dummy_by_year = data_dummy[,c(1:7,13:64)]
data_dummy_by_year$year = as.matrix(data_dummy[,8:12])%*%matrix(1:5,5,1)+2010
trees = 10

## main function
boosting_para = function(train_data, test_data){ 
  small_train_x = train_data[,c(1:6,8:56)]
  small_train_x[,c('O1','V4','V6')] = (small_train_x[,c('O1','V4','V6')] - 
                                                        matrix(colMeans(small_train_x[,c('O1','V4','V6')]), 
                                                               nrow = dim(small_train_x[,c('O1','V4','V6')])[1],
                                                               ncol = dim(small_train_x[,c('O1','V4','V6')])[2],
                                                               byrow = T))/matrix(sqrt(diag(var(small_train_x[,c('O1','V4','V6')]))),
                                                                                  nrow = dim(small_train_x[,c('O1','V4','V6')])[1],
                                                                                  ncol = dim(small_train_x[,c('O1','V4','V6')])[2],byrow = T)
  small_train_weight = train_data$I2
  small_train_y = train_data$inc
  
  small_test_weight = test_data$I2
  small_test_x = test_data[,c(1:6,8:56)]
  small_test_x[,c('O1','V4','V6')] = (small_test_x[,c('O1','V4','V6')] - 
                                                       matrix(colMeans(small_test_x[,c('O1','V4','V6')]), 
                                                              nrow = dim(small_test_x[,c('O1','V4','V6')])[1],
                                                              ncol = dim(small_test_x[,c('O1','V4','V6')])[2],
                                                              byrow = T))/matrix(sqrt(diag(var(small_test_x[,c('O1','V4','V6')]))),
                                                                                 nrow = dim(small_test_x[,c('O1','V4','V6')])[1],
                                                                                 ncol = dim(small_test_x[,c('O1','V4','V6')])[2],byrow = T)
  
  small_test_y =  test_data$inc
  
  trees = 10
  alpha = 0.5
  x = small_train_x
  y = small_train_y
  weight = small_train_weight
  prob = weight*(y/sum(y)*sum(y == 0)+(y ==0))
  test_set = small_test_x
  names(test_set) = names(small_train_x)
  test_label = small_test_y
  nround = 100
  
  output_gbt = test_label
  output_svm = test_label
  output_glm = test_label
  output_gbt_01 = test_label
  output_glm_01 = test_label
  output_svm = test_label
  output_train = NULL
  output_mlp = test_label
  
  nrows = length(y)
  for(tree in 1:trees){
    boostrap = sample(1:nrows, size = alpha*nrows, replace = T, prob = prob)
    boostrap_list = list(x[boostrap,],y[boostrap])
    params = list(max.depth = 500, eta = 0.1,
                  objective='binary:logistic', eval_metric='auc')
    train_data = xgb.DMatrix(as.matrix(boostrap_list[[1]]),label = boostrap_list[[2]])
    xgb_fit = xgb.train(params = params,
                        data = train_data,
                        nrounds = nround, verbose = 1, print.every.n = 10)
    pre_test = predict(xgb_fit, newdata = as.matrix(test_set))
    
    svm_fit = svm(x=boostrap_list[[1]], y=boostrap_list[[2]], cost = 2)
    pre_svm = predict(svm_fit, newdata = as.matrix(test_set))
    # 
    pre_train = predict(xgb_fit,newdata = as.matrix(small_train_x))
    output_train = cbind(output_train,pre_train)
    output_gbt = cbind(output_gbt,pre_test)
    output_svm = cbind(output_svm,pre_svm)
    print(tree)
    output_svm = cbind(output_svm,pre_svm)
    
    mlp_fit = mx.mlp(as.matrix(boostrap_list[[1]]),boostrap_list[[2]], hidden_node = c(100, 50), out_node = 2, 
                     num.round=200, array.batch.size=1000, learning.rate=0.03, momentum=0.9,out_activation="softmax",
                     activation = 'relu', dropout = 0.8,
                     eval.metric=mx.metric.accuracy)
    pre_mlp_trian = predict(mlp_fit, as.matrix(small_train_x))
    pre_mlp_trian_lm_fit = lm(y~., data=data.frame(y = small_train_y, t(pre_mlp_trian)))
    pre_mlp_test = predict(mlp_fit, as.matrix(small_test_x))
    preds = predict(pre_mlp_trian_lm_fit, data.frame(t(pre_mlp_test)))
    
    output_mlp = cbind(output_mlp,preds)
  }
  
  glm_data = data.frame(small_train_x, y = small_train_y)
  names(glm_data) = c(names(small_test_x),'y')
  glm_fit = glm(y~.,data = glm_data, family='binomial',
                control = list(epsilon = 1e-12, maxit = 2))
  pre_glm = predict(glm_fit, newdata = test_set, type ="response")
  pre_glm_01 = predict(glm_fit, newdata = test_set, type ="response")
  output_glm = cbind(output_glm,pre_glm)
  return(list(output_gbt, output_glm, output_mlp, output_svm))
  # return(list(output_gbt,output_svm))
}
data_list = list(data_dummy_by_year[data_dummy_by_year$year==2010,], 
                 data_dummy_by_year[data_dummy_by_year$year==2011,], 
                 data_dummy_by_year[data_dummy_by_year$year==2012,], 
                 data_dummy_by_year[data_dummy_by_year$year==2013,], 
                 data_dummy_by_year[data_dummy_by_year$year==2014,])

### list for output
output_para_list = NULL

for(i in 1:4){
  output_para_list[[i]] = boosting_para(data_list[[i]],data_list[[i+1]])
  print(paste('year',i+2010))
}

# write.csv(output_para_list[[1]], '/Users/nick/Desktop/output_freq_1.csv')
# write.csv(output_para_list[[2]], '/Users/nick/Desktop/output_freq_2.csv')
# write.csv(output_para_list[[3]], '/Users/nick/Desktop/output_freq_3.csv')
# write.csv(output_para_list[[4]], '/Users/nick/Desktop/output_freq_4.csv')

auc_gbt = NULL
auc_glm = NULL
auc_mlp = NULL
auc_svm = NULL
auc = function(yhat,y) (sum(c(1:length(y))[y[order(yhat)] == 1]) - sum(y)*(sum(y)+1)/2) / sum(y) / length(y)

for(i in 1:4){
  auc_gbt = c(auc_gbt, auc(rowMeans(output_para_list[[i]][[1]][,2:trees]), output_para_list[[i]][[1]][,1]))
  auc_glm = c(auc_glm, auc(output_para_list[[i]][[2]][,2], output_para_list[[i]][[2]][,1]))
  auc_mlp = c(auc_mlp, auc(rowMeans(output_para_list[[i]][[3]][,2:trees]), output_para_list[[i]][[3]][,1]))
  auc_svm = c(auc_svm, auc(rowMeans(output_para_list[[i]][[4]][,2:trees]), output_para_list[[i]][[4]][,1]))
}
auc_gbt
auc_glm
auc_mlp
auc_svm


ggdata = data.frame(auc = c(auc_glm,auc_gbt,auc_mlp, auc_svm), 
                    year = c(2011:2014,2011:2014,2011:2014, 2011:2014), 
                    model = c(rep('glm',4),rep('gbt',4),rep('mlp',4),rep('svm',4)))
ggplot(ggdata, aes(x=year, y=auc, fill=model)) +
  geom_bar(stat = "identity",position="dodge")+
  scale_fill_grey(start = 0.4, end = 0)+
  theme_bw()+
  geom_text(aes(x = year, y = auc, label = round(auc,3)) ,vjust = 1.5, colour = 'white',position = position_dodge(0.9),size = 3) +
  coord_cartesian(ylim=c(0.6,0.66))


##### ROC ########################################3
# roc = function(threshold, pre_vec, true_vec){
#   table = table(true_vec, pre_vec<threshold )
#   if ( dim(table)[1] == 1) table = rbind(table, c(0,0))
#   if ( dim(table)[2] == 1) table = cbind(table, c(0,0))
#   TP = table[1,1]/(table[1,1]+table[1,2])
#   FP = table[2,1]/(table[2,1]+table[2,2])
#   return(c(TP,FP))
# }
# 
# roc_gbt = NULL
# roc_glm = NULL
# roc_mlp = NULL
# roc_svm = NULL
# i=2
# for (threshold in quantile(rowMeans(output_para_list[[i]][[1]][,2:trees]),prob = seq(0,1,by=0.02))){
#   roc_gbt = rbind(roc_gbt, roc(threshold, rowMeans(output_para_list[[i]][[1]][,2:trees]), output_para_list[[i]][[1]][,1]))
# }
# for (threshold in quantile(output_para_list[[i]][[2]][,2],prob = seq(0,1,by=0.02))){
#   roc_glm = rbind(roc_glm, roc(threshold, output_para_list[[i]][[2]][,2], output_para_list[[i]][[3]][,1]))
# }
# for (threshold in quantile(rowMeans(output_para_list[[i]][[3]][,2:trees]),prob = seq(0,1,by=0.02))){
#   roc_mlp = rbind(roc_mlp, roc(threshold, rowMeans(output_para_list[[i]][[3]][,2:trees]), output_para_list[[i]][[3]][,1]))
# }
# for (threshold in quantile(rowMeans(output_para_list[[i]][[4]][,2:trees]),prob = seq(0,1,by=0.02))){
#   roc_svm = rbind(roc_svm, roc(threshold, rowMeans(output_para_list[[i]][[4]][,2:trees]), output_para_list[[i]][[4]][,1]))
# }
# 
# 
# roc_gbt = data.frame(x = roc_gbt[,1], y = roc_gbt[,2], classifier = 'gbt')
# roc_glm = data.frame(x = roc_glm[,1], y = roc_glm[,2], classifier = 'glm')
# roc_mlp = data.frame(x = roc_mlp[,1], y = roc_mlp[,2], classifier = 'mlp')
# roc_svm = data.frame(x = roc_svm[,1], y = roc_svm[,2], classifier = 'svm')
# 
# ggdata_roc = rbind(roc_gbt,roc_glm,roc_mlp,roc_svm)
# ggplot(ggdata_roc, aes(x=x, y=y, colour=classifier)) + geom_line(size = 1) +
#   scale_colour_brewer(palette="Set1")+
#   theme_bw()+
#   geom_point(size=2, shape=20)+
#   xlab('False Positive Rate')+ylab('True Positive Rate')

#########################################################

F_measure = function(threshold,pre_vec, true_vec){
  # pre : 0,1区间上的预测值
  # true : 真实的0，1
  table = table(true_vec, pre_vec > threshold )
  if ( dim(table)[1] == 1) table = rbind(c(0,0),table)
  if ( dim(table)[2] == 1) table = cbind(c(0,0),table)
  P = table[2,2]/(table[2,2]+table[1,2])
  R = table[2,2]/(table[2,2]+table[2,1])
  return(2/(1/P+1/R))
  # return(c(R, P))
}

F_gbt = NULL
F_glm = NULL
F_mlp = NULL
F_svm = NULL
for(i in 1:4){
  F_gbt[i] = F_measure(quantile(rowMeans(output_para_list[[i]][[1]][,2:trees]),0.8),rowMeans(output_para_list[[i]][[1]][,2:trees]), output_para_list[[i]][[1]][,1])
  F_glm[i] = F_measure(quantile(output_para_list[[i]][[2]][,2], 0.8),output_para_list[[i]][[2]][,2], output_para_list[[i]][[2]][,1])
  F_mlp[i] = F_measure(quantile(rowMeans(output_para_list[[i]][[3]][,2:trees]),0.8),rowMeans(output_para_list[[i]][[3]][,2:trees]), output_para_list[[i]][[3]][,1])
  F_svm[i] = F_measure(quantile(rowMeans(output_para_list[[i]][[4]][,2:trees]),0.8),rowMeans(output_para_list[[i]][[4]][,2:trees]), output_para_list[[i]][[4]][,1])
}
F_gbt
F_glm
F_mlp
F_svm

ggdata = data.frame(F1 = c(F_gbt, F_glm, F_mlp, F_svm), 
                    year = c(2011:2014,2011:2014,2011:2014, 2011:2014), 
                    model = c(rep('glm',4),rep('gbt',4),rep('mlp',4),rep('svm',4)))
ggplot(ggdata, aes(x=year, y=F1, fill=model)) +
  geom_bar(stat = "identity",position="dodge")+
  scale_fill_grey(start = 0.4, end = 0)+
  theme_bw()+
  geom_text(aes(x = year, y = F1, label = round(F1,3)) ,vjust = 1.5, colour = 'white',position = position_dodge(0.9),size = 3)+
  coord_cartesian(ylim=c(0.5,0.61))
#######################






