# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 07:32:23 2019

@author: Dustin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



###############################################################################
#PART ONE: GENERATE DATA BY SIMULATIONS {random numbers = uniform distribution over the interval [-2, +2]}
###############################################################################

#STEP 1

np.random.seed(seed = 229)
#select 16 random numbers Aij with i= 1 2 3 4 and j = 1 2 3 4
A = np.random.uniform(-2,2,size = (4,4))

#select 4 random numbers Bi with i= 1 2 3 4
B = np.random.uniform(-2,2, size = 4)

#select 1 random number c (c/20)
C = np.random.uniform(-2,2)/20

#display the values of these random numbers

#Define the polynomial of degree 2 in the 4 variables x1 x2 x3 x4 as follows

#STEP 2

#select 10, 000 vectors x1 ... x10,000 in R4
X = np.random.uniform(-2,2, size = (10000, 4))

#for each selected xn compute U(n) = Pol(xn) and y(n) = sign[U(n)]
#  Do a for loop because it was faster than Numpy
U = []
for row in range(0,10000):
    u = np.dot(np.dot(X[row,:].T , A), X[row,:]) + np.dot(X[row,:], B) + C/20
    U += [u]
#Pol(x) = ∑i ∑j Aij xi xj + ∑ i Bi xi + c/20

del row, u

#get y into 1 or -1
y = pd.Series(copy.deepcopy(U))
y[y>0] = 1; y[y<0] = -1

#turn into data frame
data = np.insert(X,4,y,axis = 1)
data = pd.DataFrame(data, columns = [0,1,2,3,'y'])

#keep only 2500 cases in CL(1) and 2500 cases in CL(-1),  
d = pd.concat([data[data['y']>0].sample(n=2500, axis = 0, random_state = 229),
                    data[data['y']<0].sample(n=2500, axis = 0, random_state = 229)]
                    , axis = 0)
d = d.reset_index(drop = True)
d.describe()
#                 0            1            2            3          y
#count  5000.000000  5000.000000  5000.000000  5000.000000  5000.0000
#mean     -0.030798    -0.006511     0.017532    -0.007001     0.0000
#std       1.151772     1.142018     1.163467     1.158793     1.0001
#min      -1.999214    -1.999030    -1.999205    -1.999118    -1.0000
#25%      -1.027578    -0.981700    -0.988848    -1.000714    -1.0000
#50%      -0.039267    -0.025135     0.017507    -0.053413     0.0000
#75%       0.965622     0.986252     1.024374     1.006964     1.0000
#max       1.999050     1.998722     1.999957     1.999582     1.0000

#Center and Rescale  this data set of size 5000  so that the standardized 
#data set will have mean = 0 and dispersion =1  
x_scale = pd.DataFrame(preprocessing.scale(d[[0,1,2,3]]))
d_scale = pd.concat([x_scale,d['y']], axis = 1)
d_scale.describe()
#                  0             1             2             3          y
#count  5.000000e+03  5.000000e+03  5.000000e+03  5.000000e+03  5000.0000
#mean   2.229245e-16  3.157474e-17 -2.613021e-16 -1.353584e-16     0.0000
#std    1.000100e+00  1.000100e+00  1.000100e+00  1.000100e+00     1.0001
#min   -1.709205e+00 -1.744909e+00 -1.733559e+00 -1.719303e+00    -1.0000
#25%   -8.655192e-01 -8.540025e-01 -8.650699e-01 -8.576272e-01    -1.0000
#50%   -7.354152e-03 -1.630917e-02 -2.168156e-05 -4.005603e-02     0.0000
#75%    8.652058e-01  8.693934e-01  8.654672e-01  8.751056e-01     1.0000
#max    1.762546e+00  1.756044e+00  1.704065e+00  1.731787e+00     1.0000

#Then Split each class into a training set and a test set , 
#using the proportions 80% (2000) and 20% (500)
test = pd.concat([d_scale[d_scale['y']>0].sample(n=500, axis = 0, random_state = 229),
                    d_scale[d_scale['y']<0].sample(n=500, axis = 0, random_state = 229)]
                    , axis = 0)
train = d_scale.drop(index = test.index.tolist())
test = test.reset_index(drop = True); train = train.reset_index(drop = True)

test.describe()
#                 0            1            2            3          y
#count  1000.000000  1000.000000  1000.000000  1000.000000  1000.0000
#mean     -0.072690    -0.018303     0.001209     0.044672     0.0000
#std       1.014888     1.004584     0.996168     0.982818     1.0005
#min      -1.701889    -1.740440    -1.724199    -1.718540    -1.0000
#25%      -0.953009    -0.849071    -0.819694    -0.782770    -1.0000
#50%      -0.133823    -0.068271     0.004469     0.016663     0.0000
#75%       0.799407     0.846853     0.829280     0.894336     1.0000
#max       1.758899     1.756044     1.703412     1.727214     1.0000

train.describe()
#                 0            1            2            3            y
#count  4000.000000  4000.000000  4000.000000  4000.000000  4000.000000
#mean     -0.005065    -0.003072     0.003298    -0.009365     0.000000
#std       0.998458     1.001684     1.000537     1.003748     1.000125
#min      -1.713265    -1.728817    -1.760184    -1.743854    -1.000000
#25%      -0.870641    -0.850352    -0.869974    -0.869782    -1.000000
#50%      -0.002689    -0.018255     0.029563    -0.055263     0.000000
#75%       0.846080     0.865216     0.867895     0.879854     1.000000
#max       1.768606     1.750326     1.724248     1.718833     1.000000

###############################################################################
#PART TWO: SVM classification by linear kernel 
###############################################################################

from SVM import SVM

mySVM = SVM()

#SVM on Training set, linear kernel
mySVM.set_kernel(kernel = 'linear')
#run the function
n_sv, sv, y_predict, score, dist_hp, model, sv_coef = \
    mySVM.run_svm(train.drop(columns = 'y'), train['y'], C=5, random_state = 9989, train = 'yes')

#Get density histrograms of accuracy
mySVM.get_hist(dist_hp, train['y'])

#get confusion matrix for the training set
train_conf, train_per_conf = mySVM.get_confusions(train['y'], y_predict)

#SVM on test set
test_predict, test_score = mySVM.run_svm(test.drop(columns = 'y'), 
                                                       test['y'], C=5, random_state = 9989, train = 'no', 
                                                       model = model)

#get confusion matrix on test set
test_conf, test_per_conf = mySVM.get_confusions(test['y'], test_predict)

#Compute the errors of estimation

###############################################################################
#PART THREE: Optimize the parameter cost
###############################################################################

#Tune the cost
mySVM = SVM()

#SVM on Training set, linear kernel
mySVM.set_kernel(kernel = 'linear')

#run through the different tunes
model, C_scores, C_n_sv = mySVM.tune_svm(train.drop(columns = 'y'), \
                                    train['y'], C=5, random_state = 9989, \
                                    tune = 'C', params = [1, 25, 75, 150, 300, 600])

#print the different scores for each c
print(max(C_scores), min(C_scores[max(C_scores)]))
best_c = min(C_scores[max(C_scores)])

mySVM = SVM()

#SVM on Training set, linear kernel
mySVM.set_kernel(kernel = 'linear')
#run the function
n_sv2, sv2, y2_predict, score2, dist_hp2, model = \
            mySVM.run_svm(train.drop(columns = 'y'), train['y'], C=best_c, 
                          random_state = 9989, train = 'yes')

#Get density histrograms of accuracy
mySVM.get_hist(dist_hp2, train['y'])

#get confusion matrix for the training set
train_conf, train_per_conf = mySVM.get_confusions(train['y'], y2_predict)

#Do test run

###############################################################################
#PART FOUR:  SVM classification by radial kernel 
###############################################################################

radSVM = SVM()

#SVM on Training set, linear kernel
radSVM.set_kernel(kernel = 'rbf', gamma = 1)

#run the function
n_svr, svr, yr_predict, score_r, dist_hp_r, model_r = \
    radSVM.run_svm(train.drop(columns = 'y'), train['y'], C=best_c, 
                  random_state = 9989, train = 'yes')

#Get density histrograms of accuracy
mySVM.get_hist(dist_hp_r, train['y'])

#get confusion matrix for the training set
train_conf_r, train_per_conf_r = mySVM.get_confusions(train['y'], yr_predict)

#SVM on test set
test_predict_r, test_score_r = mySVM.run_svm(test.drop(columns = 'y'), 
                            test['y'], C=best_c, random_state = 9989, train = 'no', 
                            model = model_r)

#get confusion matrix on test set
test_conf_r, test_per_conf_r = mySVM.get_confusions(test['y'], test_predict_r)


###############################################################################
#PART FIVE:  Optimize cost and gamma
###############################################################################

#Tune the radial gamma and cost
tune_radSVM = SVM()

#SVM on Training set, linear kernel
tune_radSVM.set_kernel(kernel = 'rbf')

#run through the different tunes
rtune_model, C_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = 'y'), \
                                    train['y'], C=5, random_state = 9989, \
                                    tune = 'gamma', params = [.00001, .001, .01, .1, .5, 1])

#print the different scores for each c
print(max(C_scores), min(C_scores[max(C_scores)]))


###############################################################################

#NUMPY WAS SLOWER
########################################
#y = copy.deepcopy(U)
#y[y>0] = 1; y[y<0] = -1
#
#time_a = time.time()
#trial = np.dot(X,np.dot(X,A).T).diagonal()
#time_b = time.time()
#check_trial = []
#for row in range(0,10000):
#    s = 0
#    for i in range(0,4):
#        for j in range(0,4):
#            calc = A[i,j]*X[row,i]*X[row,j]
#            s += calc
#    check_trial.append(s)
#
#check_trial = np.array(check_trial)
#time_c = time.time()
#
#print(time_b-time_a, time_c-time_b)
#sum(trial-check_trial)

#from sklearn import svm
#
##set up the train
#lin = svm.SVC(C = 5, kernel = 'linear', probability = True, random_state = 9989)
#
##train with data
#lin.fit(train.drop(columns = 'y'), train['y'])
#
##Predict
#train_predict = lin.predict(train.drop(columns = 'y'))
#
##Score
#score = lin.score(train.drop(columns = 'y'), train['y'])
#
##Support Vectors
#s_vectors = lin.support_vectors_
#n_s = lin.n_support_
#
##probabilities (A,B)
#probabilites = {
#        'A' : lin.probA_,
#        'B' : lin.probB_
#        }
#
##decision function (function values are proportional to the distance of the samples X to the separating hyperplane)
#dist_hp = lin.decision_function(train.drop(columns = 'y'))
#
#pair y actual and distances
#hist = pd.concat([
#        pd.Series(dist_hp2, name = 'Distance to HP'),
#        pd.Series(train['y'])
#        ], axis = 1)

##split into group one and group 2
#hist_1 = hist[hist['y']>0]
#hist_2 = hist[hist['y']<0]
#
##plot histograms
#fig1, ax1 = plt.subplots(1,2, figsize = (12,8))
#hist_1['Distance to HP'].plot.hist(subplots = True, sharey = True, sharex = False, ax = ax1[0], bins = 15, rwidth = .95, title = 'Density Plot')
#hist_2['Distance to HP'].plot.hist(subplots = True, sharey = True, sharex = False, ax = ax1[1], bins = 15, rwidth = .95)
#
##Another way to plot
#fig1, ax1 = plt.subplots(figsize = (12,8))
#ax1.hist.hist(column = 'Distance to HP', by = 'y', layout = (2,1), sharex = True, sharey = True, bins = 15, rwidth = .95)
#
##confusion matrix
#confusion = confusion_matrix(train_predict,train['y'] )
#
##percent confusion matrix
#totals = np.sum(confusion, axis = 1)
#perc_confusion = confusion/totals
#
##print confusion matrix
#print(perc_confusion)
