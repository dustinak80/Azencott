# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:42:20 2019

@author: Dustin
"""

class SVM:
    """
    playing with a class for running SVM
    Sets the kernel type and its relative parameters
    runs svm on train/test, set the general parameters there
    print histogram of class test
    get confusion matrix and percent confusion matrix
    """
    
    def __init__(self):
        #set some parameters
        self.__kernel = None
        self.__degree = 3
        self.__gamma = 'auto'
        self.__coef0 = 0
    
    def set_kernel(self, kernel, degree = 3, gamma = 'auto', coef0 = 0):
        #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            print("""\nThat kernel is not described, your options are:
                'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'\n""")
        else: 
            self.__kernel = kernel
        
        if self.__kernel in ['poly']:
            if degree > 1:
                self.__degree = degree
                print('degree: {}'.format(self.__degree))
            else:
                print('need the degree to be greater than one')
        
        if self.__kernel in ['poly', 'rbf', 'sigmoid']:
            if gamma in ['auto','scale']:
                self.__gamma = gamma
            elif type(float(gamma)) is float:
                self.__gamma = float(gamma)
                print('gamma: {}'.format(self.__gamma))
            else:
                print('Gamma needs to be corrected')
                
        if self.__kernel in ['poly', 'sigmoid']:
            if type(float(coef0)) is float:
                self.__coef0 = float(coef0)
                print('coef0: {}'.format(self.__coef0))
            else:
                print('needs to be a number')
                   
    def run_svm(self, x, y, C = 1, random_state = None, train = 'None', model = None):
        from sklearn import svm
        if train == 'yes':
            #set up
            model = svm.SVC(C, self.__kernel, degree = self.__degree, coef0 = self.__coef0, gamma = self.__gamma, random_state = random_state)
            print('\n', model, '\n')
            #train
            model.fit(x, y)
            print('fit')
            #number of support vectors, list of support vectors
            n_sv = model.n_support_
            sv = model.support_vectors_
            print('sv')
            #predict
            y_predict = model.predict(x)
            print('predict')
            #score
            score = model.score(x, y)
            print('score')
            #decision function (distance from hyperplane)
            dist_hp = model.decision_function(x)
            print('decision function\n\n')
            #Get the support vector coefficients
            sv_coeff = model.coef_
            
            return n_sv, sv, y_predict, score, dist_hp, model, sv_coeff
        
        elif train == 'no':
            #predict
            test_predict = model.predict(x)
            print('predict')
            #score
            test_score = model.score(x, y)
            print('score')
        
            return test_predict, test_score
        
        else:
            print('train needs to equal yes or no')  
    
    def tune_svm(self, x, y, C = 1, random_state = None, tune = None, params = None):
        from sklearn import svm
        
        score_tracker = {}
        n_sv_tracker = {}
        
        for i in params:
            #set up
            print(params)
            if tune == 'C':
                print('C: {}'.format(i))
                model = svm.SVC(i, self.__kernel, random_state = random_state)
            if tune == 'degree':
                self.set_kernel(self.__kernel, degree = i)
                model = svm.SVC(C, self.__kernel, degree = self.__degree, random_state = random_state)
            if tune == 'gamma':
                self.set_kernel( self.__kernel, gamma = i)
                model = svm.SVC(C, self.__kernel, gamma = self.__gamma, random_state = random_state)
            if tune == 'coef0':
                self.set_kernel(self.__kernel, coef0 = i)
                model = svm.SVC(C, self.__kernel, coef0 = self.__coef0, random_state = random_state)
            print('\n', model, '\n')
            #train
            model.fit(x, y)
            print('fit')
            #number of support vectors, list of support vectors
            n_sv = model.n_support_
            #score
            score = model.score(x, y)
            print('score')
            print(score)
            #decision function (distance from hyperplane)
            

            n_sv_tracker[i] = [n_sv]
            
            if score in score_tracker.keys():
                score_tracker[score] = score_tracker[score] + [i]
            else:
                score_tracker[score] = [i]
        
        
        return model, score_tracker, n_sv_tracker
        
    def get_hist(self, dist_hp, y):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        hist = pd.concat([
                pd.Series(dist_hp, name = 'Distance to HP'),
                pd.Series(y, name = 'y')
                ], axis = 1) 
        
        #split into group one and group 2
        hist_1 = hist[hist['y']>0]
        hist_2 = hist[hist['y']<0]
        
        ##plot histograms
        fig1, ax1 = plt.subplots(1,2, figsize = (12,8))
        hist_1['Distance to HP'].plot.hist(subplots = True, sharey = True, 
              sharex = False, ax = ax1[0], bins = 15, rwidth = .95, 
              title = 'Density Plot (+ on left, - on right)')
        hist_2['Distance to HP'].plot.hist(subplots = True, 
              sharey = True, sharex = False, ax = ax1[1], bins = 15, 
              rwidth = .95)
        
        print('+ is on left, - is on right')
    
    def get_confusions(self, y_actual, y_predict):
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        #confusion
        confusion = confusion_matrix(y_predict,y_actual )
        
        #percent confusion
        totals = np.sum(confusion, axis = 1)
        perc_confusion = confusion/totals
        
        return confusion, perc_confusion
    
    
    
    
    
    