# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:48:19 2019

@author: Dustin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time


###############################################################################
#Import Data/Describe Data
###############################################################################
fname = \
r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW4\data2.csv'
fname1 = \
r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW4\data2_appliance.csv'
data = pd.read_csv(fname)

#Class and Meanings
class_name = {0:'Low Wattage',
              1:'Medium Wattage',
              2:'High Wattage'}

#plot discrete frequncy
fig, ax = plt.subplots(figsize = (12,8))
ax.hist(data['month'], align = 'left',rwidth = .95, bins = [1,2,3,4,5,6])
ax.set_title('Frequency of Months')
ax.set_xlabel('Month')

###############################################################################
#Organize the Data for running SVM
###############################################################################

#Test and Train
import test_train as tt

#set it up
cont = tt.test_train()

#Organize Data (1st step)
cont.set_xy_class(list(data.columns), y_columns = ['Class', 'Wattage'], column = 'Class')

#get dummies
#data_final = cont.get_dummies(data, dummies = ['month'])

#print description before
data.describe()
#             Class      Wattage  ...   Visibility    Tdewpoint
#count  6579.000000  6579.000000  ...  6579.000000  6579.000000
#mean      0.974920   101.531134  ...    38.329685     3.761663
#std       0.824236    93.047128  ...    11.664370     4.194497
#min       0.000000    20.000000  ...     1.000000    -6.550000
#25%       0.000000    50.000000  ...    29.166667     0.900000
#50%       1.000000    63.333333  ...    40.000000     3.433333
#75%       2.000000   110.000000  ...    40.000000     6.566667
#max       2.000000   796.666667  ...    66.000000    15.400000

#Get standardized
data_final, stdz_params = cont.get_standardized(data, classification = True)

#print description after
data_final.describe()
#             Class      Wattage  ...    Visibility     Tdewpoint
#count  6579.000000  6579.000000  ...  6.579000e+03  6.579000e+03
#mean      0.974920   101.531134  ...  6.361634e-16 -1.927829e-16
#std       0.824236    93.047128  ...  1.000076e+00  1.000076e+00
#min       0.000000    20.000000  ... -3.200561e+00 -2.458566e+00
#25%       0.000000    50.000000  ... -7.856159e-01 -6.822942e-01
#50%       1.000000    63.333333  ...  1.432089e-01 -7.828234e-02
#75%       2.000000   110.000000  ...  1.432089e-01  6.687850e-01
#max       2.000000   796.666667  ...  2.372389e+00  2.774879e+00

#Greater than two classes
data_final = cont.greater_two(data_final)

#get test and train
train, test = cont.get_test_train(data = data_final, set_seed = 229)

###############################################################################
#SVM
###############################################################################
from SVM import SVM

mySVM = SVM()

#largest classes
data_final.groupby('Class').sum().iloc[:,0:3]
#       0_vs_all  2_vs_all  1_vs_all
#Class                              
#0          2319     -2319     -2319
#1         -2106     -2106      2106
#2         -2154      2154     -2154
class_to_num = {'SVM0' : 0,
                'SVM1' : 1,
                'SVM2' : 2
        }

train.groupby('Class', axis = 0).count()['0_vs_all']
#Class
#0    1855
#1    1684
#2    1723

test.groupby('Class', axis = 0).count()['0_vs_all']
#Class
#0    464
#1    422
#2    431

############################  Get Proportions for each class ##################
indx_0 = np.random.choice(a = np.array(train[train['Class']==0].index), 
                              size = int(.5*len(train[train['Class']==0].index)), replace = False)
indx_2 = np.random.choice(a = np.array(train[train['Class']==2].index), 
                          size = int(.5*len(train[train['Class']==2].index)), replace = False)
indx_1 = np.random.choice(a = np.array(train[train['Class']==1].index), 
                          size = int(.5*len(train[train['Class']==1].index)), replace = False)

#get the training set with 0
index_for_0 = list(indx_1)+list(indx_2) + list(train[train['Class']==0].index)
print('index:', len(index_for_0), '\  uniques:', len(np.unique(np.array(index_for_0))))
train_0 = train.iloc[index_for_0].sample(frac = 1, axis = 'index').reset_index(drop = True)
test_0 = pd.concat([test, train.drop(index_for_0, axis = 0)], axis = 0).sample(frac = 1, axis = 'index').reset_index(drop = True)
print('\nTrain 0: \n{}'.format(train_0.groupby(by = 'Class', axis = 0).count()['0_vs_all']))
print('\nTest 0: \n{}'.format(test_0.groupby(by = 'Class', axis = 0).count()['0_vs_all']))

#get the training set with 1
index_for_1 = list(indx_0)+list(indx_2) + list(train[train['Class']==1].index)
print('index:', len(index_for_1), '\  uniques:', len(np.unique(np.array(index_for_1))))
train_1 = train.iloc[index_for_1].sample(frac = 1, axis = 'index').reset_index(drop = True)
test_1 = pd.concat([test, train.drop(index_for_1, axis = 0)], axis = 0).sample(frac = 1, axis = 'index').reset_index(drop = True)
print('\nTrain 1: \n{}'.format(train_1.groupby(by = 'Class', axis = 0).count()['0_vs_all']))
print('\nTest 1: \n{}'.format(test_1.groupby(by = 'Class', axis = 0).count()['0_vs_all']))

#get the training set with 2
index_for_2 = list(indx_0)+list(indx_1) + list(train[train['Class']==2].index)
print('index:', len(index_for_2), '\  uniques:', len(np.unique(np.array(index_for_2))))
train_2 = train.iloc[index_for_2].sample(frac = 1, axis = 'index').reset_index(drop = True)
test_2 = pd.concat([test, train.drop(index_for_2, axis = 0)], axis = 0).sample(frac = 1, axis = 'index').reset_index(drop = True)
print('\nTrain 2: \n{}'.format(train_2.groupby(by = 'Class', axis = 0).count()['0_vs_all']))
print('\nTest 2: \n{}'.format(test_2.groupby(by = 'Class', axis = 0).count()['0_vs_all']))

#################  Optimize Gamma and Cost for class 0  ########################
mySVM.set_kernel(kernel = 'rbf')
time_SVM0 = {}

##Work on optimization
a = time.time()
param_grid = {'gamma': [.00001, .001, .1, 1],
              'C': [.00001, .1, 20, 60, 100]
        }

#grid search
cv, best_estimator, b_score, b_params = mySVM.gridsearch(train_0[cont._test_train__x_columns],
                train_0['0_vs_all'], param_grid = param_grid, random_state = 229,
                cv=10, return_train_score=True, it = 0.1)

time_SVM0['Grid Search'] = [time.time()-a]

# Manual Tune (gamma = .01)
a = time.time()

mySVM.set_kernel(kernel = 'rbf', gamma = .01)
tune0_scores, n_sv0 = mySVM.tune_svm(train_0[cont._test_train__x_columns], 
                                    train_0['0_vs_all'], test_0[cont._test_train__x_columns], test_0['0_vs_all'], 
                                    C=1, random_state = 229, tune = 'C', 
                                    params = [25, 30, 35, 40, 45, 50, 55], it = 0.1)

time_SVM0['Manual Tune'] = [time.time()-a]

# Manual Tune (Cost = 50)
a = time.time()

mySVM.set_kernel(kernel = 'rbf', gamma = .01)
tune0_scores, n_sv0 = mySVM.tune_svm(train_0[cont._test_train__x_columns], 
                                    train_0['0_vs_all'], test_0[cont._test_train__x_columns], test_0['0_vs_all'], 
                                    C=50, random_state = 229, tune = 'gamma', 
                                    params = [.005, .0075, .01, .025, .05, .075, .1, .125], it = 0.2)

time_SVM0['Manual Tune'] += [time.time()-a]


#################  Optimize Gamma and Cost for class 1  #######################
mySVM.set_kernel(kernel = 'rbf')
time_SVM1 = {}

##Work on optimization
a = time.time()
param_grid = {'gamma': [.00001, .001, .1, 1],
              'C': [.00001, .1, 20, 60, 100]
        }

#grid search
cv, best_estimator, b_score, b_params = mySVM.gridsearch(train_1[cont._test_train__x_columns],
                train_1['1_vs_all'], param_grid = param_grid, random_state = 229,
                cv=10, return_train_score=True, it = 1.1)

time_SVM1['Grid Search'] = [time.time()-a]

# Manual Tune (gamma = .1)
a = time.time()

mySVM.set_kernel(kernel = 'rbf', gamma = .1)
tune1_scores, n_sv1 = mySVM.tune_svm(train_1[cont._test_train__x_columns], 
                                    train_1['1_vs_all'], test_1[cont._test_train__x_columns], test_1['1_vs_all'], 
                                    C=1, random_state = 229, tune = 'C', 
                                    params = [10, 15, 17.5, 20, 22.5, 25, 30, 35], it = 1.1)

time_SVM1['Manual Tune'] = [time.time()-a]

# Manual Tune (Cost = 22.5)
a = time.time()

mySVM.set_kernel(kernel = 'rbf', gamma = .1)
tune1_scores, n_sv1 = mySVM.tune_svm(train_1[cont._test_train__x_columns], 
                                    train_1['1_vs_all'], test_1[cont._test_train__x_columns], test_1['1_vs_all'], 
                                    C=22.5, random_state = 229, tune = 'gamma', 
                                    params = [.01, .025, .05, .075, .1, .125, .150, .175], it = 1.2)

time_SVM1['Manual Tune'] += [time.time()-a]

#################  Optimize Gamma and Cost for Class 2  #####################
##Work on optimization
mySVM.set_kernel(kernel = 'rbf')
time_SVM2 = {}

#grid search
a = time.time()
param_grid = {'gamma': [.00001, .001, .1, 1],
              'C': [.00001, .1, 20, 60, 100]
        }

cv, best_estimator, b_score, b_params = mySVM.gridsearch(train_2[cont._test_train__x_columns],
                train_2['2_vs_all'], param_grid = param_grid, random_state = 229,
                cv=10, return_train_score=True, it = 2.1)

time_SVM2['Grid Search'] = [time.time()-a]

# Manual Tune (gamma = .1)
a = time.time()

mySVM.set_kernel(kernel = 'rbf', gamma = .1)
tune2_scores, n_sv2 = mySVM.tune_svm(train_2[cont._test_train__x_columns], 
                                    train_2['2_vs_all'], test_2[cont._test_train__x_columns], test_2['2_vs_all'], 
                                    C=1, random_state = 229, tune = 'C', 
                                    params = [10, 15, 17.5, 20, 22.5, 25, 30, 35], it = 2.1)

time_SVM2['Manual Tune'] = [time.time()-a]

# Manual Tune (Cost = 20)
a = time.time()

mySVM.set_kernel(kernel = 'rbf', gamma = .1)
tune2_scores, n_sv2 = mySVM.tune_svm(train_2[cont._test_train__x_columns], 
                                    train_2['2_vs_all'], test_2[cont._test_train__x_columns], test_2['2_vs_all'], 
                                    C=20, random_state = 229, tune = 'gamma', 
                                    params = [.01, .025, .05, .075, .1, .125, .150, .175], it = 2.2)

time_SVM2['Manual Tune'] += [time.time()-a]


###############################################################################
#3 SVM's, 3 Confusion Matrixes/sets of histograms
###############################################################################

###########      SVM 0 vs All (0 is low wattage)  #############################
a = time.time()

#set Kernel
gamma = .025
C = 50
svm0 = SVM()
svm0.set_kernel(kernel = 'rbf', gamma = gamma)

#training set 
r_sv0, sv0, y_predict0, score0, conf_int0, dist_hp0, model0 = \
    svm0.run_svm(train_0[cont._test_train__x_columns], train_0['0_vs_all'], C=C, random_state = 229)

train_conf0, train_per_conf0, train_limits0, train_confidence0 = svm0.get_confusions(train_0['0_vs_all'], y_predict0)

#histogram
hist_dist0 = svm0.get_hist(dist_hp0, train_0['0_vs_all'], it = 0.1)

time_SVM0['Train'] = [time.time()-a]
a = time.time()

#test set
test_predict0, test_score0, test_conf_int0, test_dist_hp0  = \
    svm0.run_svm(test_0[cont._test_train__x_columns], test_0['0_vs_all'], C=C, random_state = 229,
                  train = 'no', model = model0)

test_conf0, test_per_conf0, test_limits0, test_confidence0 = svm0.get_confusions(test_0['0_vs_all'], test_predict0, conf_desired = .95)

#histogram
test_hist_dist0 = svm0.get_hist(test_dist_hp0, test_0['0_vs_all'], it = 0.2)

time_SVM0['Test'] = [time.time()-a]
###########      SVM 1 vs All (1 is med wattage)  #############################
a = time.time()

#set Kernel
gamma = .05
C = 22.5
svm1 = SVM()
svm1.set_kernel(kernel = 'rbf', gamma = gamma)

#fit/training set
r_sv1, sv1, y_predict1, score1, conf_int1, dist_hp1, model1 = \
    svm1.run_svm(train_1[cont._test_train__x_columns], train_1['1_vs_all'], C=C, random_state = 229)

train_conf1, train_per_conf1, train_limits1, train_confidence1 = svm1.get_confusions(train_1['1_vs_all'], y_predict1)

#histogram
hist_dist1 = svm1.get_hist(dist_hp1, train_1['1_vs_all'], it = 1.1)

time_SVM1['Train'] = [time.time()-a]
a = time.time()

#test set
test_predict1, test_score1, test_conf_int1, test_dist_hp1  = \
    svm1.run_svm(test_1[cont._test_train__x_columns], test_1['1_vs_all'], C=C, random_state = 229,
                  train = 'no', model = model1)

test_conf1, test_per_conf1, test_limits1, test_confidence1 = svm1.get_confusions(test_1['1_vs_all'], test_predict1, conf_desired = .95)

#histogram
test_hist_dist1 = svm1.get_hist(test_dist_hp1, test_1['1_vs_all'], it = 1.2)

time_SVM1['Test'] = [time.time()-a]
###########      SVM 2 vs All (2 is high wattage)  #############################
a = time.time()

#set Kernel
gamma = .075
C = 20
svm2 = SVM()
svm2.set_kernel(kernel = 'rbf', gamma = gamma)

#fit
n_sv2, sv2, y_predict2, score2, conf_int2, dist_hp2, model2 = \
    svm2.run_svm(train_2[cont._test_train__x_columns], train_2['2_vs_all'], C=C, random_state = 229)

train_conf2, train_per_conf2, train_limits2, train_confidence2 = svm2.get_confusions(train_2['2_vs_all'], y_predict2)

#histogram
hist_dist2 = svm2.get_hist(dist_hp2, train_2['2_vs_all'], it = 2.1)

time_SVM2['Train'] = [time.time()-a]
a = time.time()

#test set
test_predict2, test_score2, test_conf_int2, test_dist_hp2  = \
    svm2.run_svm(test_2[cont._test_train__x_columns], test_2['2_vs_all'], C=C, random_state = 229,
                  train = 'no', model = model2)

test_conf2, test_per_conf2, test_limits2, test_confidence2 = svm2.get_confusions(test_2['2_vs_all'], test_predict2, conf_desired = .95)

#histogram
test_hist_dist2 = svm2.get_hist(test_dist_hp2, test_2['2_vs_all'], it = 2.2)

time_SVM2['Test'] = [time.time()-a]
###############################################################################
#4 SVM's, 3 Confusion Matrixes/sets of histograms
###############################################################################

"""
Make a train and test table
run each svm's distances through each svms cdf
put them next to each other
pick the best one 
"""

#Get all of your predictions from full training set and full test set
#SVM 0
y_predict0, score0, conf_int0, dist_hp0= \
    svm0.run_svm(train[cont._test_train__x_columns], train['0_vs_all'], C=C, random_state = 229,
                 train = 'no', model = model0)

test_predict0, test_score0, test_conf_int0, test_dist_hp0  = \
    svm0.run_svm(test[cont._test_train__x_columns], test['0_vs_all'], C=C, random_state = 229,
                  train = 'no', model = model0)

#SVM 1
y_predict1, score1, conf_int1, dist_hp1= \
    svm1.run_svm(train[cont._test_train__x_columns], train['1_vs_all'], C=C, random_state = 229,
                 train = 'no', model = model1)

test_predict1, test_score1, test_conf_int1, test_dist_hp1  = \
    svm1.run_svm(test[cont._test_train__x_columns], test['1_vs_all'], C=C, random_state = 229,
                  train = 'no', model = model1)

#SVM 2
y_predict2, score2, conf_int2, dist_hp2= \
    svm2.run_svm(train[cont._test_train__x_columns], train['2_vs_all'], C=C, random_state = 229,
                 train = 'no', model = model2)

test_predict2, test_score2, test_conf_int2, test_dist_hp2  = \
    svm2.run_svm(test[cont._test_train__x_columns], test['2_vs_all'], C=C, random_state = 229,
                  train = 'no', model = model2)


#split up the groups
train_class = pd.DataFrame(np.zeros((train.shape[0],4)), columns = ['SVM0', 'SVM1', 'SVM2', 'Prediction'])
train_class['SVM0'] = hist_dist0.cdf(dist_hp0)
train_class['SVM1'] = hist_dist1.cdf(dist_hp1)
train_class['SVM2'] = hist_dist2.cdf(dist_hp2)

train_class['Prediction']=train_class.idxmax(axis = 1)
train_class['Prediction'].replace(to_replace = class_to_num, inplace = True)

test_class = pd.DataFrame(np.zeros((test.shape[0],4)), columns = ['SVM0', 'SVM1', 'SVM2', 'Prediction'])
test_class['SVM0'] = hist_dist0.cdf(test_dist_hp0)
test_class['SVM1'] = hist_dist1.cdf(test_dist_hp1)
test_class['SVM2'] = hist_dist2.cdf(test_dist_hp2)

test_class['Prediction']=test_class.idxmax(axis = 1)
test_class['Prediction'].replace(to_replace = class_to_num, inplace = True)

#Get confusion Matrix
final_test_conf, final_test_per_conf, final_test_limits, final_test_confidence = svm2.get_confusions(test['Class'], test_class['Prediction'], conf_desired = .95)
final_train_conf, final_train_per_conf, final_train_limits, final_train_confidence = svm2.get_confusions(train['Class'], train_class['Prediction'], conf_desired = .95)

###############################################################################
#TEST/NOT USED
###############################################################################

##Train SVM1 (1_vs_all) with a portion of each each of other classes not all of them
#    #to see if proportionality makes a difference
#indx_0 = np.random.choice(a = np.array(train[train['Class']==0].index), 
#                              size = int(.5*len(train[train['Class']==0].index)), replace = False)
#indx_2 = np.random.choice(a = np.array(train[train['Class']==2].index), 
#                          size = int(.5*len(train[train['Class']==2].index)), replace = False)
#indx_1 = np.random.choice(a = np.array(train[train['Class']==1].index), 
#                          size = int(.5*len(train[train['Class']==1].index)), replace = False)
#train_t_indx = list(indx_0)+list(indx_1)+list(indx_2)
#
#train_t = train.iloc[train_t_indx,:]
#
#gamma = .0001#1*10**(-5)
#C = 95
#svm2 = SVM()
#svm2.set_kernel(kernel = 'rbf', gamma = gamma)
#
##training set
#n_sv2, sv2, y_predict2, score2, conf_int2, dist_hp2, model2 = \
#    svm2.run_svm(train_t[cont._test_train__x_columns], train_t['1_vs_all'], C=C, random_state = 229)
#
#train_conf2, train_per_conf2, train_limits2, train_confidence2 = svm2.get_confusions(train_t['1_vs_all'], y_predict2)
#
##Get Plots
##histogram
#svm2.get_hist(dist_hp2, train_t['1_vs_all'])
#
##test set
#test_predict2, test_score2, test_conf_int2, test_dist_hp2  = \
#    svm2.run_svm(test[cont._test_train__x_columns], test['1_vs_all'], C=C, random_state = 229,
#                  train = 'no', model = model2)
#
#test_conf2, test_per_conf2, test_limits2, test_confidence2 = svm2.get_confusions(test['1_vs_all'], test_predict2, conf_desired = .95)
#
##Get Plots
##histogram
#svm2.get_hist(test_dist_hp2, test['1_vs_all'])



##Testing test train split
#testin = data.copy(deep = True)
#testin['class2'] = 0
#testin['class2'][testin['Wattage']>100]='two'
#testin.iloc[0:20,0] = 5
#testin.iloc[40:50, 0] = 'five'
#train, test = tt_.get_test_train(data = testin, dummies = ['month','class2'], column = 'Class', set_seed = 229)
