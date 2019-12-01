# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:49:57 2019

@author: Dustin
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

fname = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW4\KAG_energydata_complete.csv'
data = pd.read_csv(fname)

###############################################################################
#Explor Data - try to seperate between weekday, weekend
###############################################################################
# 01-11-2016 = Monday
# '2016-01-11 17:00:00'
date = str(list(data['date']))
month = list(map(int,re.findall('-([0-9].)-', date)))
day = list(map(int,re.findall('-([0-9].)\s', date)))
hour = list(map(int,re.findall('\s([0-9].):', date)))
minutes = list(map(int,re.findall(':([0-9].):', date)))

#insert new columns into dataframe, drop date column, get column list
data.insert(1, 'month', month)
data.insert(2, 'day', day)
data.insert(3, 'hour', hour)
data.insert(4, 'minute', minutes)

data.drop('date', axis = 1, inplace = True)

features = list(data.columns)

#change minute values to 0,30
data['minute'][data['minute']<30] = 0
data['minute'][data['minute']>30] = 30

data['minute'].unique()



#test
by_min = {}
count = 0
for i in list(data['month'].unique()):
    for j in list(data['day'][data['month']==i].unique()):
        for k in list(data['hour'][data['day'] == j].unique()):
            for l in list(data['minute'][data['hour']==k].unique()):
                #Meeting conditions of month, day, hour, minutes and getting mean
                by_min[count] = data[data['month'] == i][data['day'] == j]\
                [data['hour'] == k][data['minute']==l].mean()
                #key names for dictionary
                count += 1

#get into data frame
data_index = pd.DataFrame(by_min).T
data_plot = pd.DataFrame(by_min).T
#data_changed.drop('date', axis = 1, inplace = True)
data_index.dropna(axis = 0, inplace = True)
data_plot.dropna(axis = 0, inplace = True)
data_index = data_index[features]
data_plot = data_plot[features]

#change indices to month and day
data_index.set_index(['month', 'day','hour','minute'], drop = False, inplace = True)

#Create a new column total wattage
#wattage = list(data_index['Appliances'] + data_index['lights'])
wattage = list(data_index['Appliances'])
data_index.insert(0, 'Wattage', wattage)
data_plot.insert(4, 'Wattage', wattage)

#drop some columns for index
data_index.drop(['Appliances', 'lights', 'minute', 'rv1', 'rv2'], axis = 1, inplace = True)

#Get average temperature inside and get difference from outside
#Temperature six was outside on north side
inside_temp = data_index[['T1','T2','T3','T4','T5','T7','T8','T9']]
total_inside = inside_temp.sum(axis = 1)/8
#average of outside - two temperatures
outside_temp = data_index[['T6','T_out']]
total_outside = outside_temp.sum(axis = 1)/2
#get the difference
difference = abs(total_outside - total_inside)
data_index.insert( 22, 'T_inside', total_inside)
data_index.insert(23, 'T_outside', total_outside)
data_index.insert( 23, 'T_difference', difference)
#data_index['T_difference']=difference

#get some data by index
monthly_means = data_index.mean(level = 0)
monthly_means.reset_index(drop = True, inplace = True)
daily_means = data_index.mean(level = [0,1])
daily_means.reset_index(drop = True, inplace = True)

fig, ax = plt.subplots(figsize=(12,8))
daily_means.boxplot(column = 'Wattage', by = 'month', ax = ax)

fig, ax = plt.subplots(figsize=(12,8))
daily_means.boxplot(column = 'Wattage', by = 'T_difference', ax = ax)

#Plot Wattage vs t difference
srtd = daily_means[['Wattage','T_difference']].sort_values(by = 'T_difference')
fig1, ax1 = plt.subplots(figsize=(12,8))
srtd.plot('T_difference', 'Wattage', kind = 'line', ax = ax1)
ax1.set_ylabel('Wattage')
ax1.set_title('Wattage vs T_difference')

#PCA for plotting 3d
x = data_index.drop(columns = ['Class', 'Wattage'])
corr = x.corr()
eigs, vecs = np.linalg.eig(corr)
rat_eig = eigs/sum(eigs)
cum_sum = rat_eig.cumsum()
cum_sum[2]
#fist 3 projections
data_proj = pd.DataFrame(np.dot(x,vecs[:,0:3]))
#get index's
data_rindex = data_index.reset_index(drop=True)
index0 = data_rindex[data_rindex['Class']==0].index
index1 = data_rindex[data_rindex['Class']==1].index
index2 = data_rindex[data_rindex['Class']==2].index

from mpl_toolkits import mplot3d 

fig2 = plt.figure(figsize = (12,8))
ax2 = fig2.add_subplot(111, projection = '3d')
ax2.view_init(60,10)
ax2.set_title('1st 3 Eigenvector Projections \n({} variance explained)'.format(round(cum_sum[2],2)))
ax2.set_xlabel('Principal Component 1') 
ax2.set_ylabel('Principal Component 2') 
ax2.set_zlabel('Principal Component 3') 
ax2.scatter(data_proj.iloc[index0,0], data_proj.iloc[index0,1], data_proj.iloc[index0,2], c='r' , alpha = .2) 
ax2.scatter(data_proj.iloc[index1,0], data_proj.iloc[index1,1], data_proj.iloc[index1,2], c='b' , alpha = .2) 
ax2.scatter(data_proj.iloc[index2,0], data_proj.iloc[index2,1], data_proj.iloc[index2,2], c='g' , alpha = .2) 
ax2.legend(['Low Wattage','Med Wattage','High Wattage']) 


#describe data
data_index['Wattage'].describe()
quantiles = list(data_index['Wattage'].quantile(q = [.33,.66]))
#0.33    53.333333
#0.66    90.000000

group_1 = data_index[data_index['Wattage'] <= quantiles[0]]
group_1.insert(0, 'Class', 0)
group_2 = data_index[data_index['Wattage'] > quantiles[0]][data_index['Wattage'] <= quantiles[1]]
group_2.insert(0, 'Class', 1)
group_3 = data_index[data_index['Wattage'] > quantiles[1]]
group_3.insert(0, 'Class', 2)

data_index.insert(0, 'Class', 0)
data_index['Class'][data_index['Wattage'] > quantiles[0]] = 1
data_index['Class'][data_index['Wattage'] > quantiles[1]] = 2

group_1.describe()
#           Wattage        month  ...          rv1          rv2
#count  2319.000000  2319.000000  ...  2319.000000  2319.000000
#mean     45.674860     2.816731  ...    25.117109    25.117109
#std       7.605298     1.369127  ...     8.378420     8.378420
#min      20.000000     1.000000  ...     1.730898     1.730898
#25%      43.333333     2.000000  ...    19.153694    19.153694
#50%      46.666667     3.000000  ...    25.063295    25.063295
#75%      50.000000     4.000000  ...    31.080565    31.080565
#max      53.333333     5.000000  ...    48.163597    48.163597
group_2.describe()
#           Wattage        month  ...          rv1          rv2
#count  2319.000000  2319.000000  ...  2319.000000  2319.000000
#mean     45.674860     2.816731  ...    25.117109    25.117109
#std       7.605298     1.369127  ...     8.378420     8.378420
#min      20.000000     1.000000  ...     1.730898     1.730898
#25%      43.333333     2.000000  ...    19.153694    19.153694
#50%      46.666667     3.000000  ...    25.063295    25.063295
#75%      50.000000     4.000000  ...    31.080565    31.080565
#max      53.333333     5.000000  ...    48.163597    48.163597
group_3.describe()
#           Wattage        month  ...          rv1          rv2
#count  2319.000000  2319.000000  ...  2319.000000  2319.000000
#mean     45.674860     2.816731  ...    25.117109    25.117109
#std       7.605298     1.369127  ...     8.378420     8.378420
#min      20.000000     1.000000  ...     1.730898     1.730898
#25%      43.333333     2.000000  ...    19.153694    19.153694
#50%      46.666667     3.000000  ...    25.063295    25.063295
#75%      50.000000     4.000000  ...    31.080565    31.080565
#max      53.333333     5.000000  ...    48.163597    48.163597

#WRITE IT TO CSV
data_index.to_csv('data2_appliance.csv', index = False)

table_columns = pd.concat([pd.Series(list(range(1,len(data_index.columns)+1))), pd.Series(data_index.columns)], axis = 1)

###############################################################################


##try using index, didnt work
#data.loc[(1,11),:]
#data.loc[(1,11), data['hour'] == 17]
#data[:,data.loc[(1,11),'hour'] == 17]
#data[:,data.loc[(1,11),'hour'] == 17]

## Go through list and get mean of values that share same half hour
##IS THERE A BETTER AND FASTER WAY TO DO THIS??????
#by_hr = {}
#count = 0
#for i in list(data['month'].unique()):
#    for j in list(data['day'][data['month']==i].unique()):
#        for k in list(data['hour'][data['day'] = j].unique()):
#            for l in list(data['minute'].unique()):
#                #Meeting conditions of month, day, hour, minutes and getting mean
#                by_hr[count] = data[data['month'] == i][data['day'] == j]\
#                [data['hour'] == k][data['minute']==l].mean()
#                #key names for dictionary
#                count += 1
#        
#data_changed = pd.DataFrame(by_hr)