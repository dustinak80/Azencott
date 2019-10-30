# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:40:55 2019

@author: Dustin
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

nl='\n'

#Import Data
auto_o= pd.read_csv('auto.csv')

"""PRELIMINARY TREATMENT OF DATA SET"""
#discard last three items
auto = auto_o[['mpg','cylinders','displacement','horsepower','weight','acceleration']]

#find and remove rows without values and change type of value
values_m = auto[auto['horsepower']=='?']
values_m = values_m.index
auto = auto.drop(values_m, axis=0)
auto = auto.astype('float64')

#features and Number of Cases
print(auto.info(), nl)
print('There are',auto.shape[0], 'Cases and', auto.shape[1], 'features.', nl)
N = len(auto)

#rename features
features = ['cyl','dis','hor','wei','acc']
auto.columns = ['mpg']+features
auto_f = auto[['cyl','dis','hor','wei','acc']]

"""compute mean and standard deviation of each feature"""
stats = pd.DataFrame()
stats['mean'] = auto.mean()
stats['std_dev'] = auto.std()
mean_std = round(stats[['mean','std_dev']],2)

"""display the histogram of each feature + mpg"""
pd.DataFrame.hist(auto, grid=True, figsize=(12,8), bins='auto', rwidth=0.9, color='#607c8e', align='left')                  
plt.savefig('histograms.png')

"""
#Individual Histograms
plt.figure(figsize = (12,8))
plt.grid(b=True, axis = 'y', )
plt.ylabel('Count')
plt.xlabel('MPG')
plt.title('MPG')
plt.hist(auto['mpg'], bins='auto', color = 'r', rwidth=0.9)

plt.figure(figsize = (12,8))
plt.grid(b=True, axis = 'y', )
plt.ylabel('Count')
plt.xlabel('CYL')
plt.title('CYL')
plt.hist(auto['cyl'], bins='auto', color = 'r', rwidth=1, align = 'left')
"""

"""display 5 following scatterplots (cyl,mpg) (dis,mpg) (hor,mpg) (wei,mpg) (acc,mpg)"""
colors = 'blue'
fsz=(12,8)

#(cyl,mpg)
plt.figure(figsize=fsz)
plt.scatter(auto['cyl'], auto['mpg'], c=colors, alpha=0.5)
plt.title('Cylinders vs MPG')
plt.xlabel('Cylinders')
plt.ylabel('MPG')

#(dis,mpg)
plt.figure(figsize=fsz)
plt.scatter(auto['dis'], auto['mpg'], c=colors, alpha=0.5)
plt.title('Displacement vs MPG')
plt.xlabel('Displacement')
plt.ylabel('MPG')

#(hor,mpg)
plt.figure(figsize=fsz)
plt.scatter(auto['hor'], auto['mpg'], c=colors, alpha=0.5)
plt.title('Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')

#(wei,mpg)
plt.figure(figsize=fsz)
plt.scatter(auto['wei'], auto['mpg'], c=colors, alpha=0.5)
plt.title('Weight vs MPG')
plt.xlabel('Weight')
plt.ylabel('MPG')

#(acc,mpg)
plt.figure(figsize=fsz)
plt.scatter(auto['acc'], auto['mpg'], c=colors, alpha=0.5)
plt.title('Acceleration vs MPG')
plt.xlabel('Acceleration')
plt.ylabel('MPG')

"""Compute the 5 Correlations"""
#Centralizing Data
auto_c = pd.DataFrame()
auto_c['mpg'] = auto['mpg'] - stats.at['mpg','mean']
auto_c['cyl'] = auto['cyl'] - stats.at['cyl','mean']
auto_c['dis'] = auto['dis'] - stats.at['dis','mean']
auto_c['hor'] = auto['hor'] - stats.at['hor','mean']
auto_c['wei'] = auto['wei'] - stats.at['wei','mean']
auto_c['acc'] = auto['acc'] - stats.at['acc','mean']
print(auto_c.mean(), nl, auto_c.sum(), nl)

#Calculating Variance and Covariance
# formula --> var_mpg = sum(auto_c['mpg']**2)/N
stats['var'] = auto.var()
stats.loc['cyl','cov'] = sum(auto_c['cyl']*auto_c['mpg'])/N
stats.loc['dis','cov'] = sum(auto_c['dis']*auto_c['mpg'])/N
stats.loc['hor','cov'] = sum(auto_c['hor']*auto_c['mpg'])/N
stats.loc['wei','cov'] = sum(auto_c['wei']*auto_c['mpg'])/N
stats.loc['acc','cov'] = sum(auto_c['acc']*auto_c['mpg'])/N

#compute Correlation --> cov(x,y)/(sd(x)*sd(y)) **note: should be between -1 and 1
stats.loc['cyl','cor'] = stats.loc['cyl','cov']/(stats.loc['cyl','std_dev']*stats.loc['mpg','std_dev'])
stats.loc['dis','cor'] = stats.loc['dis','cov']/(stats.loc['dis','std_dev']*stats.loc['mpg','std_dev'])
stats.loc['hor','cor'] = stats.loc['hor','cov']/(stats.loc['hor','std_dev']*stats.loc['mpg','std_dev'])
stats.loc['wei','cor'] = stats.loc['wei','cov']/(stats.loc['wei','std_dev']*stats.loc['mpg','std_dev'])
stats.loc['acc','cor'] = stats.loc['acc','cov']/(stats.loc['acc','std_dev']*stats.loc['mpg','std_dev'])

"""Compute Covariance Matrix and Correlation Matrix of the 5 features"""
cov_m = auto_f.cov()
corr_m = auto_f.corr()

"""Compute the 5 EigenValues. Verify they sum up to 5. Compute Ri = (L1+L2+...+Li)/n |i=1-->n , n=5"""
#eigenvalues and eigenvectors
eigs = np.linalg.eig(corr_m)
eig_val = eigs[0]
eig_vec = eigs[1]

#Verify eigenvalues sum up to 5
print(sum(eig_val), nl)

#Compute Ri
r = []
for i in range(5):
    ri = sum(eig_val[0:(i+1)])/(5)
    r.append(ri)
    i = i + 1
print(r, nl)
""" This is known as the cumulative scree plot"""

"""Reset the rows: LOWmpg < median(mpg), HIGHmpg > median(mpg). """
#ReOrder the Rows
auto_s = auto.sort_values('mpg')

#split the table in two by the median of mpg
mpg_med = auto['mpg'].median(axis=0)
LOWmpg = auto_f[auto_s['mpg'] < mpg_med]
HIGHmpg = auto_f[auto_s['mpg'] > mpg_med]

""" Print histograms of the 5 Features and determine which histgram has a good capacity to discriminate """
#Cylinders
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = fsz)
ax1.hist(LOWmpg['cyl'], bins = 'auto', rwidth = 0.9, color='green')
ax1.set_title('Low MPG')
ax1.set_xlabel('Cylinders')
ax1.set_ylabel('Qty')
ax2.hist(HIGHmpg['cyl'], bins = 'auto', rwidth = 0.9, color='blue')
ax2.set_title('High MPG')
ax2.set_xlabel('Cylinders')

#Displacement
#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = fsz)
plt.hist(LOWmpg['dis'], bins = 9, rwidth = 0.9, color='green', alpha = 0.5, stacked = True)
#ax1.set_title('Low MPG')
#ax1.set_xlabel('Displacement')
#ax1.set_ylabel('Qty')
plt.hist(HIGHmpg['dis'], bins = 9, rwidth = 0.9, color='blue', alpha = 0.5, stacked = True)
#ax2.set_title('High MPG')
#ax2.set_xlabel('Displacement')

#Horsepower
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = fsz)
ax1.hist(LOWmpg['hor'], bins = 'auto', rwidth = 0.9, color='green')
ax1.set_title('Low MPG')
ax1.set_xlabel('Horsepower')
ax1.set_ylabel('Qty')
ax2.hist(HIGHmpg['hor'], bins = 'auto', rwidth = 0.9, color='blue')
ax2.set_title('High MPG')
ax2.set_xlabel('Horsepower')

#Weight
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = fsz)
ax1.hist(LOWmpg['wei'], bins = 'auto', rwidth = 0.9, color='green')
ax1.set_title('Low MPG')
ax1.set_xlabel('Weight')
ax1.set_ylabel('Qty')
ax2.hist(HIGHmpg['wei'], bins = 'auto', rwidth = 0.9, color='blue')
ax2.set_title('High MPG')
ax2.set_xlabel('Weight')

#Acceleration
#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = fsz)
plt.hist(LOWmpg['acc'], bins = 'auto', rwidth = 0.9, color='green', stacked = True, density = True, alpha = 0.5)
#ax1.set_title('Low MPG')
#ax1.set_xlabel('Acceleration')
#ax1.set_ylabel('Qty')
plt.hist(HIGHmpg['acc'], bins = 'auto', rwidth = 0.9, color='blue', stacked = True, density = True, alpha = 0.5)
#ax2.set_title('High MPG')
#ax2.set_xlabel('Acceleration')

""" Compute mean, std dev for LOWmpg and Highmpg then compute (m_high-m_low)/s(f)"""
#Create Databases
LOWstats = pd.DataFrame()
HIGHstats = pd.DataFrame()
#Get mean and Standard Dev
LOWstats['mean'] = LOWmpg.mean()
LOWstats['std_dev'] = LOWmpg.std()
HIGHstats['mean'] = HIGHmpg.mean()
HIGHstats['std_dev'] = HIGHmpg.std()
#Calculate s(f) = (stdhigh+stdlow)/sqrt(N)
s_f = (LOWstats['std_dev']+HIGHstats['std_dev'])/np.sqrt(N)
s_f = s_f.rename('sf')
#compute (m_high-m_low)/s(f)
unk = np.abs(HIGHstats['mean']-LOWstats['mean'])/s_f#.transpose()
unk = unk.rename('unk')