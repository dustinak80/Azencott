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
import time


timea = time.time()
times = [time.time()]
###############################################################################
#PART ONE: GENERATE DATA BY SIMULATIONS {random numbers = uniform distribution over the interval [-2, +2]}
###############################################################################

#STEP 1

np.random.seed(seed = 229)
#np.random.seed(seed = 313)
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

u = pd.Series(U)
print('\npositive:', len(u[u>0]))
print('negative:', len(u[u<0]))

#get y into 1 or -1
y = pd.Series(copy.deepcopy(U))
y[y>0] = 1; y[y<0] = -1

#turn into data frame
data = np.insert(X,4,[y,U],axis = 1)
data = pd.DataFrame(data, columns = [0,1,2,3,'y','U'])

#keep only 2500 cases in CL(1) and 2500 cases in CL(-1),  
d = pd.concat([data[data['y']>0].sample(n=2500, axis = 0, random_state = 229),
                    data[data['y']<0].sample(n=2500, axis = 0, random_state = 229)]
                    , axis = 0)
d = d.reset_index(drop = True)
print('\nGreater than 0 for y, U:', len(d[d['y']>0]),len(d[d['U']>0]), '\n',
      'Less than 0 for y, U:', len(d[d['y']<0]),len(d[d['U']<0]))
U_prescale = pd.Series(d['U'])
#d.describe()
#                 0            1  ...          y            U
#count  5000.000000  5000.000000  ...  5000.0000  5000.000000
#mean     -0.032034    -0.012180  ...     0.0000    -0.641322
#std       1.148448     1.148988  ...     1.0001     4.693225
#min      -1.999433    -1.998371  ...    -1.0000   -18.668459
#25%      -1.030560    -0.992777  ...    -1.0000    -3.403314
#50%      -0.031247    -0.021481  ...     0.0000    -0.000209
#75%       0.946375     0.984573  ...     1.0000     2.273674
#max       1.998916     1.998722  ...     1.0000    13.594242

#Center and Rescale  this data set of size 5000  so that the standardized 
#data set will have mean = 0 and dispersion =1  
#d_scale = pd.DataFrame(preprocessing.scale(d), columns = [0,1,2,3,'y','U'])
d_scale = pd.DataFrame(preprocessing.scale(d), columns = [0,1,2,3,'y','U'])
print('\nGreater than 0 for y, U:', len(d_scale[d_scale['y']>0]),len(d_scale[d_scale['U']>0]), '\n',
      'Less than 0 for y, U:', len(d_scale[d_scale['y']<0]),len(d_scale[d_scale['U']<0]))
#do not carry scaled U because the mean isnt centralized, it is a skewed frequency
d_scale['U'] = U_prescale
x = d_scale[[0,1,2,3]]
y = d_scale['y']
#d_scale['U'] = U_prescale
odescription = d.describe()
description = d_scale.describe()
#                  0             1  ...          y            U
#count  5.000000e+03  5.000000e+03  ...  5000.0000  5000.000000
#mean  -1.980624e-16  3.219647e-17  ...     0.0000    -0.641322
#std    1.000100e+00  1.000100e+00  ...     1.0001     4.693225
#min   -1.713265e+00 -1.728817e+00  ...    -1.0000   -18.668459
#25%   -8.695433e-01 -8.535299e-01  ...    -1.0000    -3.403314
#50%    6.850186e-04 -8.095595e-03  ...     0.0000    -0.000209
#75%    8.520248e-01  8.675927e-01  ...     1.0000     2.273674
#max    1.768606e+00  1.750326e+00  ...     1.0000    13.594242

#Then Split each class into a training set and a test set , 
#using the proportions 80% (2000) and 20% (500)
test = pd.concat([d_scale[d_scale['y']>0].sample(n=500, axis = 0, random_state = 229),
                    d_scale[d_scale['y']<0].sample(n=500, axis = 0, random_state = 229)]
                    , axis = 0)
train = d_scale.drop(index = test.index.tolist())
test = test.reset_index(drop = True); train = train.reset_index(drop = True)

#Plot two elements of the data
#first get highest correlation
d_scale[[0,1,2,3,'y']].corr()['y']

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(x = d_scale[[0]][d_scale['y']>0], y = d_scale[[2]][d_scale['y']>0],
            color = 'blue')
plt.scatter(x = d_scale[[0]][d_scale['y']<0], y = d_scale[[2]][d_scale['y']<0],
            color = 'red')
plt.title('Two dimensional View of Data')
plt.xlabel('Feature 0')
plt.ylabel('Feature 2')

times += [time.time()-timea]
timea = time.time()
###############################################################################
#PART TWO: SVM classification by linear kernel 
###############################################################################

from SVM import SVM

mySVM = SVM()

#SVM on Training set, linear kernel
mySVM.set_kernel(kernel = 'linear')
#run the function
n_sv, sv, y_predict, score, conf_int, dist_hp, model = \
    mySVM.run_svm(train.drop(columns = ['y','U']), train['y'], C=5, random_state = 9989)

#Get density histrograms of accuracy on train
mySVM.get_hist(dist_hp, train['y'])

#plot polynomial vs. svm
mySVM.get_scatter(train['U'], dist_hp)

#get confusion matrix for the training set
train_conf, train_per_conf, train_limits, train_confidence = mySVM.get_confusions(train['y'], y_predict, conf_desired = .95)

#SVM on test set
test_predict, test_score, test_conf_int, test_dist_hp  = \
    mySVM.run_svm(test.drop(columns = ['y','U']), test['y'], C=5, random_state = 9989,
                  train = 'no', model = model)

#plot polynomial vs. svm
mySVM.get_scatter(test['U'], test_dist_hp)

#get confusion matrix on test set and limits
test_conf, test_per_conf, test_limits, test_confidence = mySVM.get_confusions(test['y'], test_predict, conf_desired = .95)

#look at these two values to verify what column means what in confusion matrix
print('\nValue Counts in response:\n', pd.Series(test_predict).value_counts())
print('\nSum of columns in conf matrix:',test_conf.sum(axis = 0))

###############################################################################
#PART THREE: Optimize the parameter cost
###############################################################################

#Tune the cost
mySVM = SVM()
    
#SVM on Training set, linear kernel
mySVM.set_kernel(kernel = 'linear')
    
#run through the different tunes: #1
C_scores, C_n_sv = mySVM.tune_svm(train.drop(columns = ['y','U']), \
                                 train['y'], test.drop(columns = ['y','U']), test['y'], \
                                 C=5, random_state = 9989, tune = 'C', \
                                 params = [.00001, .001, 1, 10, 30, 50])

#run through the different tunes
C_scores, C_n_sv = mySVM.tune_svm(train.drop(columns = ['y','U']), \
                                 train['y'], test.drop(columns = ['y','U']), test['y'], \
                                 C=5, random_state = 9989, tune = 'C', \
                                 params = [.001, .01, .1, 1, 1.5, 2])

    
##print the different scores for each c
#print(max(C_scores.values()), C_scores.keys()[max(C_scores)]))
best_c = .1
    
mySVM = SVM()
    
#SVM on Training set, linear kernel
mySVM.set_kernel(kernel = 'linear')
#run the function
n_sv, sv, y_predict, score, conf_int, dist_hp, model = \
            mySVM.run_svm(train.drop(columns = ['y','U']), train['y'], C=best_c, 
                          random_state = 9989)
    
#Get density histrograms of accuracy
mySVM.get_hist(dist_hp, train['y'])
    
#get confusion matrix for the training set
train_conf, train_per_conf, train_limits, train_confidence = mySVM.get_confusions(train['y'], y_predict, conf_desired = .95)
 
#plot polynomial vs. svm
mySVM.get_scatter(train['U'], dist_hp)
   
#Do test run
#SVM on test set
test_predict, test_score, test_conf_int, test_dist_hp  = \
    mySVM.run_svm(test.drop(columns = ['y','U']), test['y'], C=5, random_state = 9989,
                  train = 'no', model = model)

#plot polynomial vs. svm
mySVM.get_scatter(test['U'], test_dist_hp)

#get confusion matrix on test set and limits
test_conf, test_per_conf, test_limits, test_confidence = mySVM.get_confusions(test['y'], test_predict, conf_desired = .95)

times += [time.time()-timea]
timea = time.time()
###############################################################################
#PART FOUR:  SVM classification by radial kernel 
###############################################################################

radSVM = SVM()

#SVM on Training set, linear kernel
radSVM.set_kernel(kernel = 'rbf', gamma = 1)

#run the function
n_svr, svr, yr_predict, score_r, conf_r, dist_hp_r, model_r = \
    radSVM.run_svm(train.drop(columns = ['y', 'U']), train['y'], C=best_c, 
                  random_state = 9989, train = 'yes')

#Get density histrograms of accuracy
radSVM.get_hist(dist_hp_r, train['y'])

#get confusion matrix for the training set
train_conf_r, train_per_conf_r, train_limits, train_confidencer = radSVM.get_confusions(train['y'], yr_predict, conf_desired = .95)

#plot polynomial vs. svm
radSVM.get_scatter(train['U'], dist_hp_r)

#SVM on test set
test_predict_r, test_score_r, test_conf_r, test_dist_hp_r = radSVM.run_svm(test.drop(columns = ['y', 'U']), 
                            test['y'], C=best_c, random_state = 9989, train = 'no', 
                            model = model_r)

#get confusion matrix on test set
test_conf_r, test_per_conf_r, test_limitsr, test_confidencer = radSVM.get_confusions(test['y'], test_predict_r, conf_desired = .95)

#plot polynomial vs. svm
radSVM.get_scatter(test['U'], test_dist_hp_r)

###############################################################################
#PART FIVE:  Optimize cost and gamma
###############################################################################

#Tune the radial gamma and cost
tune_radSVM = SVM()

#SVM on Training set, linear kernel
tune_radSVM.set_kernel(kernel = 'rbf')

#run through the different tunes - first iteration
gamma_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=best_c, random_state = 9989, tune = 'gamma', 
                                    params = [.00001, .001, .01, .1, .5, 1])

#run through the different tunes - 2nd iteration
gamma_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y','U']), test['y'], 
                                    C=best_c, random_state = 9989, tune = 'gamma', 
                                    params = [.05, .075, .1, .25, .45, .65, .75])

"""
looks like gamma of .1 is the best for both Test and Train Set \
Use gamma = 0.45 and iterate through the cost
"""

tune_radSVM = SVM()

#SVM on Training set, linear kernel
tune_radSVM.set_kernel(kernel = 'rbf', gamma = .45)

C_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=5, random_state = 9989, tune = 'C', 
                                    params = [.00001, .01, 10, 35, 70, 100])

C_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=5, random_state = 9989, tune = 'C', 
                                    params = [.00001, .001, 1, 10, 30, 50])

C_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=5, random_state = 9989, tune = 'C', 
                                    params = [15, 20, 25, 30, 35, 40, 45])


"""
Looks like cost of 35 was the highest and closest match between test and train \
We will re-iterate through gamma using a cost of 30, with focus around previous \
gamma.
"""

tune_radSVM = SVM()

#SVM on Training set
tune_radSVM.set_kernel(kernel = 'rbf')

#run through the different tunes with best cost to make sure didnt change
gamma_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=35, random_state = 9989, tune = 'gamma', 
                                    params = [.05, .075, .1, .25, .45, .65, .75])

#SVM on Training set, linear kernel
tune_radSVM.set_kernel(kernel = 'rbf', gamma = .1)
C_scores, n_sv_gamma = tune_radSVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=5, random_state = 9989, tune = 'C', 
                                    params = [8, 15, 22, 30, 37, 45, 50])

"""
Run: gamma = 0.1, C = 35
"""

#run tuned radial SVM
C=35; gamma = 0.10

tune_radSVM = SVM()

#SVM on Training set
tune_radSVM.set_kernel(kernel = 'rbf', gamma = gamma)

n_svrt, svrt, yrt_predict, score_rt, conf_int_rt, dist_hp_rt, model_rt = \
    tune_radSVM.run_svm(train.drop(columns = ['y', 'U']), train['y'], C=C, 
                  random_state = 9989)

#Get density histrograms of accuracy
tune_radSVM.get_hist(dist_hp_rt, train['y'])

#get confusion matrix for the training set
train_conf_rt, train_per_conf_rt, limits_rt, train_confidence_rt = tune_radSVM.get_confusions(train['y'], yrt_predict)

#plot polynomial vs. svm
tune_radSVM.get_scatter(train['U'], dist_hp_rt)

#SVM on test set
test_predict_rt, test_score_rt, test_conf_int_rt, test_dist_hp_rt = tune_radSVM.run_svm(test.drop(columns = ['y', 'U']), 
                            test['y'], C=C, random_state = 9989, train = 'no', 
                            model = model_rt)

#get confusion matrix on test set
test_conf_rt, test_per_conf_rt, test_limits_rt, test_confidence_rt = tune_radSVM.get_confusions(test['y'], test_predict_rt)

#plot polynomial vs. svm
tune_radSVM.get_scatter(test['U'], test_dist_hp_rt)

times += [time.time()-timea]
timea = time.time()
###############################################################################
#PART SIX:  Run Polynomial with degree = 4, optimize Coeff and Cost
###############################################################################

"""
Since we know degree is 4, lets get an initial look with default parameters
"""

poly_SVM = SVM()

#set polynomial
poly_SVM.set_kernel('poly', degree = 4)

#run poly SVM
n_svp, svp, yp_predict, score_p, conf_p, train_dist_hp_p, model_p = \
    poly_SVM.run_svm(train.drop(columns = ['y', 'U']), train['y'], C = 1,
                  random_state = 9989, train = 'yes')
    
#Get density histrograms of accuracy
poly_SVM.get_hist(train_dist_hp_p, train['y'])

#get confusion matrix for the training set
train_conf_p, train_per_conf_p, train_limits_p, train_confidence_p = poly_SVM.get_confusions(train['y'], yp_predict)

#plot polynomial vs. svm
poly_SVM.get_scatter(train['U'], train_dist_hp_p)

#SVM on test set
test_predict_p, test_score_p, test_conf_p, test_dist_hp_p = poly_SVM.run_svm(test.drop(columns = ['y', 'U']), 
                            test['y'], C=1, random_state = 9989, train = 'no', 
                            model = model_p)

#get confusion matrix on test set
test_conf_p, test_per_conf_p, test_limits_p, test_confidence_p = poly_SVM.get_confusions(test['y'], test_predict_p)

poly_SVM.get_scatter(test['U'], test_dist_hp_p)

"""
Tune the parameters Coeff and Cost. Leave Degree = 4
"""
tpoly_SVM = SVM()

#set polynomial
tpoly_SVM.set_kernel('poly', degree = 4)

pt_scores, n_sv_pt = tpoly_SVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=1, random_state = 9989, tune = 'coef0', 
                                    params = [.001, .1, 10, 30, 60, 100, 200])

tpoly_SVM = SVM()

#set polynomial
tpoly_SVM.set_kernel('poly', degree = 4)

pt_scores, n_sv_pt = tpoly_SVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=1, random_state = 9989, tune = 'coef0', 
                                    params = [15, 20, 25, 30, 35, 40, 45])

"""
Coef of 30 and 50 were equally best on Test. 30 was 100% on Train, 50 was 99.975%.\
Use Coef 30, change params for Cost now.
"""
tpoly_SVM = SVM()

#set polynomial
tpoly_SVM.set_kernel('poly', degree = 4, coef0 = 30)

pt_scores, n_sv_pt = tpoly_SVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=1, random_state = 9989, tune = 'C', 
                                    params = [.001, .1, 25, 50, 75, 100, 125])

tpoly_SVM = SVM()

#set polynomial
tpoly_SVM.set_kernel('poly', degree = 4, coef0 = 30)

pt_scores, n_sv_pt = tpoly_SVM.tune_svm(train.drop(columns = ['y', 'U']), 
                                    train['y'], test.drop(columns = ['y', 'U']), test['y'], 
                                    C=1, random_state = 9989, tune = 'C', 
                                    params = [.1, 1, 5, 10, 15, 20, 25])

"""
Looks like best cost is 1, though they are so close it is not really different.\
Use Cost = 1, coef0 = 30, degree = 4 and run_SVM
"""
tpoly_SVM = SVM()

#set polynomial
tpoly_SVM.set_kernel('poly', degree = 4, coef0 = 30)

#run poly SVM
n_svpt, svpt, ypt_predict, score_pt, train_conf_pt, dist_hp_pt, model_pt = \
    tpoly_SVM.run_svm(train.drop(columns = ['y', 'U']), train['y'], C = 1,
                  random_state = 9989, train = 'yes')
    
#Get density histrograms of accuracy
tpoly_SVM.get_hist(dist_hp_pt, train['y'])

#get confusion matrix for the training set
train_conf_pt, train_per_conf_pt, train_intervals_pt, train_confidence_pt = tpoly_SVM.get_confusions(train['y'], ypt_predict)

#plot polynomial vs. svm
tpoly_SVM.get_scatter(train['U'], dist_hp_pt)

#SVM on test set
test_predict_pt, test_score_pt, test_conf_pt, test_dist_hp_pt = tpoly_SVM.run_svm(test.drop(columns = ['y', 'U']), 
                            test['y'], C=1, random_state = 9989, train = 'no', 
                            model = model_pt)

#get confusion matrix on test set
test_conf_pt, test_per_conf_pt, test_intervals_pt, test_confidence_pt = poly_SVM.get_confusions(test['y'], test_predict_pt)

#plot polynomial vs. svm
tpoly_SVM.get_scatter(test['U'], test_dist_hp_pt)

times += [time.time()-timea]
timea = time.time()
###############################################################################
#Time
###############################################################################

print(times)




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
