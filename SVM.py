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
        print('\nSet Kernel\n')

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
                   
    def run_svm(self, x, y, C = 1, random_state = None, train = 'yes', model = None, conf_desired = .95):
        from sklearn import svm
        from math import sqrt
        print('\nRun SVM\n')
        
        if train == 'yes':
            #set up
            model = svm.SVC(C, self.__kernel, degree = self.__degree, coef0 = self.__coef0, gamma = self.__gamma, random_state = random_state)
            print('\n', model, '\n')
            #train
            model.fit(x, y)
            print('fit')
            #number of support vectors, list of support vectors
            n_sv = model.n_support_
            r_sv = sum(n_sv)/len(x)
            sv = model.support_vectors_
            print('sv')
            #decision function
            dist_hp = model.decision_function(x)
            #predict
            y_predict = model.predict(x)
            print('predict')
            #score
            score = model.score(x, y)
            print('score')
            print(score)
            #confidence interval of score
            size = len(y)
            sigma = round(sqrt((score*(1-score))/size),3)
            if conf_desired == .90:
                z = 1.64
                conf = (score - z*sigma, score+z*sigma)
            if conf_desired == .95:
                z = 1.96
                conf = (score - z*sigma, score+z*sigma)
            
            return r_sv, sv, y_predict, score, conf, dist_hp, model
        
        elif train == 'no':
            #predict
            test_predict = model.predict(x)
            print('predict')
            #score
            test_score = model.score(x, y)
            print('score')
            print(test_score)
            #Get distance
            dist_hp = model.decision_function(x)
            print('distance')
            #confidence interval of score
            print('Confidence Interval')
            size = len(y)
            sigma = round(sqrt((test_score*(1-test_score))/size),3)
            if conf_desired == .90:
                z = 1.64
                conf = (test_score - z*sigma, test_score+z*sigma)
            if conf_desired == .95:
                z = 1.96
                conf = (test_score - z*sigma, test_score+z*sigma)
        
            return test_predict, test_score, conf, dist_hp
        
        else:
            print('train needs to equal yes or no')  
    
    def tune_svm(self, x, y, x_test, y_test, C = 1, random_state = None, tune = None, params = None, it = None):
        from sklearn import svm
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print('\nTuning\n')

        
        train_score_tracker = {}
        test_score_tracker = {}
        n_sv_tracker = {}
        
        for i in params:
            #set up
            print('\nNew Iteration:', params)
            if tune == 'C':
                print('C: {}'.format(i))
                model = svm.SVC(i, self.__kernel, degree = self.__degree, gamma = self.__gamma,
                                coef0 = self.__coef0, random_state = random_state)
            if tune == 'degree':
                self.set_kernel(self.__kernel, degree = i, gamma = self.__gamma, coef0 = self.__coef0)
                model = svm.SVC(C, self.__kernel, degree = self.__degree, gamma = self.__gamma,
                                coef0 = self.__coef0, random_state = random_state)
            if tune == 'gamma':
                self.set_kernel( self.__kernel, gamma = i, degree = self.__degree, coef0 = self.__coef0)
                model = svm.SVC(C, self.__kernel, degree = self.__degree, gamma = self.__gamma,
                                coef0 = self.__coef0, random_state = random_state)
            if tune == 'coef0':
                self.set_kernel(self.__kernel, coef0 = i, degree = self.__degree, gamma = self.__gamma)
                model = svm.SVC(C, self.__kernel, degree = self.__degree, gamma = self.__gamma,
                                coef0 = self.__coef0, random_state = random_state)
            print('\n', model, '\n')
            #train
            model.fit(x, y)
            print('fit')
            #number of support vectors, list of support vectors
            n_sv = model.n_support_
            #score
            train_score = model.score(x, y)
            print('train score:', train_score)
            #decision function (distance from hyperplane)

            #score for test set
            test_score = model.score(x_test,y_test)
            print('test score:', test_score)
            
            n_sv_tracker[i] = n_sv
            train_score_tracker[i] = train_score
            test_score_tracker[i] = test_score
        
        #Plot ratio of number of support vectors
        print('Plotting support vectors and Scores')
        n_sv_tracker = pd.DataFrame(n_sv_tracker, index = [-1,1]).T
        n_sv_tracker['ratio'] = n_sv_tracker.sum(axis=1)/len(y)
        
        fig3, ax3 = plt.subplots(2,1, figsize = (12,12))
        ax3[0].plot(range(len(n_sv_tracker)), n_sv_tracker.sum(axis = 1)/len(x), 'o-', color = 'orange', alpha = .5)
        ax3[0].set_xticks(np.arange(0,len(n_sv_tracker),1))
        ax3[0].set_xticklabels(list(n_sv_tracker.index))
        ax3[0].set_xlabel('{}'.format(tune))
        ax3[0].set_ylabel('Ratio of Support Vectors')
        ax3[0].set_title('Ratio of Support Vectors per {} Parameter'.format(tune))
#        plt.savefig('Ratio of Support Vectors(tune {}).png'.format(tune))
        
        print('Ratio of support vectors:\n', n_sv_tracker.sum(axis = 1)/len(x))
        
        #data frame for plotting test vs train scores
#        print('Plotting Scores')
        scores = pd.DataFrame([train_score_tracker, test_score_tracker], 
                              index = ['Train', 'Test']).T
        
#        fig2, ax2 = plt.subplots(figsize = (12,8))
        ax3[1].plot(range(len(scores)), scores['Train'], 'o--', color = 'blue', alpha = .5)
        ax3[1].plot(range(len(scores)), scores['Test'], 's--', color = 'red', alpha = .5)
        ax3[1].set_xticks(np.arange(0,len(scores),1))
        ax3[1].set_xticklabels(list(scores.index))
        ax3[1].set_xlabel('{}'.format(tune))
        ax3[1].set_ylabel('Score')
        ax3[1].set_title('Score per {} Parameter'.format(tune))
        ax3[1].legend(['Train','Test'])
        fig3.suptitle("Test vs. Train (tune {})".format(tune), fontsize=12)
#        plt.show()
        if it != None:
            plt.savefig('Score (tune {}) - {}.png'.format(tune, it))

#        scores.plot(figsize = (12,8), style = 'o--', xticks = list(scores.index),
#                    title = 'Test vs Train Scores with different {} parameters'.format(tune))
#        plt.savefig('Score(tune).png')
        return scores, n_sv_tracker
        
    def gridsearch(self, train, train_y, param_grid, iid = False, 
                   random_state = None, cv = 3, return_train_score = False, it = None):
        """
        Use gridsearch to optimize parameters
        param_grid: grid to search parameters by
        iid: T/F
        """
        from sklearn import svm
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import GridSearchCV
        import matplotlib.pyplot as plt
        import math

        #set model
        model = svm.SVC(kernel = self.__kernel, random_state = random_state)
        
        #run gridsearch
        grid_search =  GridSearchCV(model, param_grid=param_grid, iid=iid, 
                                    cv=cv, return_train_score=return_train_score)
        
        #fit
        print('fit')
        grid_search.fit(train, train_y)
        
        #cv results
        cv = grid_search.cv_results_
        
        #start plotting
        df = pd.concat([pd.DataFrame(cv['params']),
                        pd.concat([pd.Series(cv['mean_train_score'], name = 'Train'),
                                   pd.Series(cv['mean_test_score'], name = 'Test')], 
                        axis = 1)], axis = 1)
        
        print(df)
        print('Plotting')
        
        #plot
        key_lengths = {}
        count = 0
        for j in [len(param_grid[i]) for i in list(param_grid.keys())]:
            key_lengths[j] = list(param_grid.keys())[count]
            count += 1
        min_param = key_lengths[min(key_lengths)] 
        max_param = key_lengths[max(key_lengths)]
        width = max(8*min(key_lengths)/2,8)
        rows = math.ceil(min(key_lengths)/2)
        columns = 1 if math.ceil(min(key_lengths)/2) < 2 else 2
        fig, axs = plt.subplots(rows,columns, figsize = ((4/3)*width,width))
        
        if rows>1:
            a = 0
            b=0
            for j in param_grid[key_lengths[min(key_lengths)]]:
                axs[a,b].plot(range(len(df[max_param][df[min_param]==j])), 
                   df['Train'][df[min_param]==j], 'o--', color = 'blue', alpha = .5)
                axs[a,b].plot(range(len(df[max_param][df[min_param]==j])), 
                   df['Test'][df[min_param]==j], 'o--', color = 'red', alpha = .5)
                axs[a,b].set_xticks(np.arange(0,max(key_lengths),1))
                axs[a,b].set_xticklabels(list(df[max_param][df[min_param]==j]))
                axs[a,b].set_xlabel(max_param)
                axs[a,b].set_title('{} {}'.format(min_param, j))
                axs[a,b].set_ylabel('Score')
                axs[a,b].legend(['Train','Test'])
                if b == 1:
                    a = 1
                    b = 0
                    continue
                b += 1
        if rows==1 and columns > 1:
            b=0
            for j in param_grid[key_lengths[min(key_lengths)]]:
                axs[b].plot(range(len(df[max_param][df[min_param]==j])), 
                   df['Train'][df[min_param]==j], 'o--', color = 'blue', alpha = .5)
                axs[b].plot(range(len(df[max_param][df[min_param]==j])), 
                   df['Test'][df[min_param]==j], 'o--', color = 'red', alpha = .5)
                axs[b].set_xticks(np.arange(0,max(key_lengths),1))
                axs[b].set_xticklabels(list(df[max_param][df[min_param]==j]))
                axs[b].set_xlabel(max_param)
                axs[b].set_title('{} {}'.format(min_param, j))
                axs[b].set_ylabel('Score')
                axs[b].legend(['Train','Test'])
                b += 1        
        if columns==1:
            j = param_grid[key_lengths[min(key_lengths)]]
            axs.plot(range(len(df[max_param][df[min_param].isin(j)])), 
                   df['Train'][df[min_param].isin(j)], 'o--', color = 'blue', alpha = .5)
            axs.plot(range(len(df[max_param][df[min_param].isin(j)])), 
                   df['Test'][df[min_param].isin(j)], 'o--', color = 'red', alpha = .5)
            axs.set_xticks(np.arange(0,max(key_lengths),1))
            axs.set_xticklabels(list(df[max_param][df[min_param].isin(j)]))
            axs.set_xlabel(max_param)
            axs.set_title('{} {}'.format(min_param, j))
            axs.set_ylabel('Score')
            axs.legend(['Train','Test'])
        fig.suptitle("Test vs. Train by {}'s".format(min_param), fontsize=12)
        plt.show()
        if it != None:
            plt.savefig('Grid Search {}.png'.format(it))

        return cv, grid_search.best_estimator_, \
                grid_search.best_score_, grid_search.best_params_        
    
    def get_hist(self, dist_hp, y, it = None):
        import pandas as pd
        import matplotlib.pyplot as plt
        import math as math
        
        print('\nHistogram\n')
        
        hist = pd.concat([
                pd.Series(dist_hp, name = 'Distance to HP'),
                pd.Series(y, name = 'y')
                ], axis = 1) 
        
        #split into group one and group 2
        hist_1 = hist[hist['y']>0]
        hist_2 = hist[hist['y']<0]
        print('Group +1 length: {}'.format(len(hist_1)))
        print('Group -1 length: {}'.format(len(hist_2)))
        print('Total len of y: {}'.format(len(dist_hp)))
        print('Sum of both groups: {}'.format(len(hist_1)+len(hist_2)))
        
        ##plot histograms
        fig1, ax1 = plt.subplots(1,2, figsize = (16,8))
        #+1 plot
        ax1[0].hist(x = hist_1['Distance to HP'], bins = int(math.sqrt(len(hist_1))), rwidth = 1)
        ax1[0].set_title('Density Plot (+1)')
        ax1[0].set_xlabel('Distance from Hyperplane')
        ax1[0].axvline(color = 'red')
        #-1 plot
        ax1[1].hist(x = hist_2['Distance to HP'], bins = int(math.sqrt(len(hist_2))),rwidth = 1)
        ax1[1].set_title('Density Plot (-1)')
        ax1[1].set_xlabel('Distance from Hyperplane')
        ax1[1].axvline(color = 'red')
        fig1.suptitle('Frequency plots of Distance from Hyperplane', fontsize = 12)
        if it != None:
            plt.savefig('Density Plot {}.png'.format(it))
        
        ##get density of correctly classified - sum of area should be same as % conf
        print('Percent Correct +1: {}'.format(len(hist_1[hist_1['Distance to HP']>=0])/len(hist_1)))
        print('Percent Correct -1: {}'.format(len(hist_2[hist_2['Distance to HP']<=0])/len(hist_2)))
        hist_1_correct = hist_1[hist_1['Distance to HP']>=0]
        hist_2_correct = hist_2[hist_2['Distance to HP']<=0]

        ##plot histograms
        fig2, ax2 = plt.subplots( figsize = (12,8))
        #+1 plot
        ax2.hist(x = hist_1_correct['Distance to HP'], bins = int(math.sqrt(len(hist_1_correct))), rwidth = 1, color = 'green')
        ax2.set_title('Density Plot (+1)')
        ax2.set_xlabel('Distance from Hyperplane')
        ax2.axvline(color = 'red')
#        #-1 plot
#        ax2[1].hist(x = hist_2_correct['Distance to HP'], bins = int(math.sqrt(len(hist_2_correct))),rwidth = 1, color = 'green')
#        ax2[1].set_title('Density Plot (-1)')
#        ax2[1].set_xlabel('Distance from Hyperplane')
#        ax2[1].axvline(color = 'red')
#        fig2.suptitle('Frequency plots of Correct Distance from Hyperplane', fontsize = 12)
        if it != None:
            plt.savefig('Density Plot of Corrects {}.png'.format(it))
        
        #get cdf from histogram
        import scipy.stats as ss
        import numpy as np
        
        histo = np.histogram(hist_1_correct['Distance to HP'], bins = int(math.sqrt(len(hist_1_correct))))
        hist_dist_1 = ss.rv_histogram(histo)
        #hist_dist_1.cdf(2.6)
        
        return hist_dist_1

    
    def get_confusions(self, y_actual, y_predict, conf_desired = .95):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        import pandas as pd
        import numpy as np
        from math import sqrt
        print('\nConfusion\n')
        
        #confusion
        confusion = confusion_matrix(y_actual, y_predict)
        
        #percent confusion
        totals = np.sum(confusion, axis = 1)
        print(totals)
        perc_confusion = np.zeros((len(totals),len(totals)))
        for i in range(len(totals)):
            perc_confusion[i,:] = confusion[i,:]/totals[i]
        
        #Get classification report
        #print(classification_report(y_actual, y_predict, labels = target_names))
        
        #Get percent confidence for correct
        size = len(y_actual)
        print('size:', size)
        perc_corrects = np.append(np.diag(perc_confusion),
                                  (sum(np.diag(perc_confusion))/len(np.diag(perc_confusion))))
        
        #Get sigmas for perc_corrects
        a = (perc_corrects*(1-perc_corrects))/size
        sigmas = np.array([sqrt(i) for i in a])
        print('sigmas are: ', sigmas) 
        if conf_desired == .90:
            z = 1.64
            intervals = np.array([z*i for i in sigmas])
            print('Intervals are:', intervals)
            upper = perc_corrects + intervals
            lower = perc_corrects - intervals
        if conf_desired == .95:
            z = 1.96
            intervals = np.array([z*i for i in sigmas])
            print('Intervals are:', intervals)
            upper = perc_corrects + intervals
            lower = perc_corrects - intervals
        
        #create table of limits
        limits = pd.DataFrame({
        'Lower Limit': lower,
        'Percent Correct': perc_corrects,
        'Upper Limit': upper
        })
        
        perc_confusion=perc_confusion*100
        limits = limits*100
        
        return confusion, perc_confusion, limits, intervals
    
    def get_scatter(self, poly_x, y_predict):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print('\nGet Scatter Plot\n')
        #print(poly_x, y_predict)
        
        #seperate sectioins of graph
        together = pd.concat([pd.Series(poly_x, name = 'U'), pd.Series(y_predict, name = 'dist')], axis = 1)
        print('length of all 4 quadrants: {}'.format(len(together)))
        p1 = together[together['U']>=0][together['dist']>=0]
        p2 = together[together['U']<=0][together['dist']<=0]
        n1 = together[together['U']>=0][together['dist']<=0]
        n2 = together[together['U']<=0][together['dist']>=0]
        
        print('p1:', len(p1), '\n',
              'p2:', len(p2), '\n',
              'n1:', len(n1), '\n',
              'n2:', len(n2), '\n',
              )
        
        fig, ax = plt.subplots(figsize = (12,8))
        plt.scatter(x = p1['U'], y = p1['dist'], color = 'blue', alpha = 0.5)
        plt.scatter(x = p2['U'], y = p2['dist'], color = 'blue', alpha = 0.5)
        plt.scatter(x = n1['U'], y = n1['dist'], color = 'red', alpha = 0.5)
        plt.scatter(x = n2['U'], y = n2['dist'], color = 'red', alpha = 0.5)
        plt.title('Polynomial(X) vs. SVM(X)')
        plt.ylabel('SVM(X)')
        plt.xlabel('Polynomial(X)')
        plt.axhline(color = 'black')
        plt.axvline(color = 'black')
        plt.text(x = max(together['U']), y = max(together['dist']), fontweight = 'black', 
             s='Correct prediction of +1', horizontalalignment = 'right')
        plt.text(x = max(together['U']), y = min(together['dist']), fontweight = 'black', 
             s='Actual +1, Predicted -1', horizontalalignment = 'right')
        plt.text(x = min(together['U']), y = min(together['dist']), fontweight = 'black', 
             s='Correct prediction of -1', horizontalalignment = 'left')
        plt.text(x = min(together['U']), y = max(together['dist']), fontweight = 'black', 
             s='Actual -1, Predicted +1', horizontalalignment = 'left')
        plt.savefig('poly vs dist.png')
    
    
    
    
    
    