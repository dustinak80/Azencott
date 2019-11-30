# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:29:00 2019

@author: Dustin
"""

class test_train:
    
    def __init__(self):
        #set some params
        self.__train = None
        self.__test = None
        self.__column = None
        self.__x_columns = None
        self.__y_columns = None
        self.__dums_columns = None
    
    def set_xy_class(self, data_columns, y_columns, column):
        """
        data_columns: list of all the data columns
        y_columns: y-columns to attach to model for future work
        column: classification column
        NOTE:
        """
        
        #st the y columns
        self.__y_columns = y_columns
        
        #set the x_columns
        x_columns = [x for x in data_columns if x not in y_columns]
        self.__x_columns = x_columns
        
        self.__column = column
    
    def get_dummies(self, data, dummies):
        """
        Data = data to get dummie variables for
        dummies = columns to get dummie variables for
        """
        import pandas as pd
        
        data = data.copy(deep = True)
        #create dummy variables
        for i in dummies:
            data[i] = pd.Categorical(data[i])
        print(data[dummies].describe(include = 'all'))
        #get list of column names to carry as attribute
        dums = pd.get_dummies(data[dummies],prefix = dummies)
        dums_columns = list(dums.columns)
        #combine data sets
        data_dummy = pd.concat([data, dums], axis = 1)
        data_dummy.drop(columns = dummies, inplace = True)
        #change the attribute for x_columns
        self.__x_columns = [x for x in list(data_dummy.columns) if x not in self.__y_columns]
        #updated dums attribute
        self.__dums_columns = dums_columns
        
        return data_dummy

    def get_standardized(self, data, classification = False, dummies = False, get_params = True, use_params = False, params = None):
        """
        data = data to standardize
        classification = True or false (dont want to standardize y if true)
        dummies = T/F, dont want to standardize dummy variables
        get_params = mean and standard deviation for standardization if not training set
        use_params = use given params to standardize the data
        params = params to use
        """
        from sklearn import preprocessing
        import pandas as pd
        import numpy as np
        
        if classification == True:
            y = data[self.__y_columns]
            x = data[self.__x_columns]
            print('Classification')
        else:
            x= data
        
        if dummies == True:
            print('Dummies')
            dums = x[self.__dums_columns]
            x_less = [i for i in self.__x_columns if i not in self.__dums_columns]
            x = x[x_less]
        
        #get the parameters to use for future scaling
        if get_params == True:
            x_scale = x.copy(deep = True)
            means = x.mean()
            sds = x.std()
            mean_sd = pd.concat([means,sds],axis = 1).rename(columns = {0:'mean', 1:'sd'})
            for i in list(x.columns):
                mean = mean_sd.loc[i,'mean']
                sd = mean_sd.loc[i,'sd']      
                x_scale[i] = (x[i]-mean)/sd
            print('Got mean and std for each column')
        
        #use paramaters or not
        if use_params > 0:
            mean_sd = params.copy(deep = True)
            x_scale = x.copy(deep = True)
            for i in list(x.columns):
                mean = mean_sd.loc[i,'mean']
                sd = mean_sd.loc[i,'sd']      
                x_scale[i] = (x[i]-mean)/sd
            print('Scaled')
        else:
            x_scale = preprocessing.scale(x)
            print('Scaled')

            
        #what to concat together
        if classification == True and dummies == True:
            data_s = pd.concat([y, pd.DataFrame(x_scale, columns = x_less), dums], axis = 1)
            print('y, x, and dummies concat')
        elif classification == True:
            data_s = pd.concat([y, pd.DataFrame(x_scale, columns = self.__x_columns)], axis = 1)
            print('y and x concat')
        elif dummies == True:
            data_s = pd.concat([pd.DataFrame(x_scale, columns = x_less), dums], axis = 1)
            print('data and dummies concate')
        else:
            data_s = pd.DataFrame(x_scale, columns = self.__y_columns+self.__x_columns)
            print('Data no splitting')
        
        if get_params == True:
            return data_s, mean_sd
        else:
            return data_s
    
    def greater_two(self, data):
        """
        data: data to add the extra columns for seperate classes
        ***This is mostly for SVM when you can only do two classes at a time***
        """
        import pandas as pd
        
        data = data.copy(deep=True)
        #get the unique classes
        classes = list(data[self.__column].unique())
        #start adding the classes with unique features
        new_ys = []
        for i in classes:
            new_ys += ['{}_vs_all'.format(i)]
            data.insert(0, '{}_vs_all'.format(i), -1)
            data['{}_vs_all'.format(i)][data[self.__column]==i] = 1
        
        #update y_columns attribute
        self.__y_columns += new_ys
        
        return data    
        
    def get_test_train(self, data, ratio = .80, set_seed = None):
        """
        Get test and train data
        data = data to split
        ratio = ratio of split for train
        column = if classification column to split by, should be a name
        dummies = if yes then get dummy variables [column names]
        set_seed = None - give value if you want
        """
        import numpy as np
        import pandas as pd
                
        data = data.copy(deep = True)
              
        if self.__column == None:
            print('Need to set this up for regression')
        else:
            #get original lengths
            lengths = pd.Series(data.groupby(self.__column).count().iloc[:,0], name = 'lengths')
            #get index values for split
            index = {}
            spot = 0
            for i in list(lengths.index):
                index[i]= list(data[data[self.__column]==i].sample(n = int(ratio*lengths[i]), 
                     random_state = set_seed, axis = 0).index)
                spot += 1
            #make datasets
            indices = []
            for i in list(index.keys()):
                indices += index[i]
            #train
            train = data.iloc[indices,:].sort_index()
            train = train.reset_index(drop = True)
            self.__train = train
            #test
            test = data.drop(index = indices).sort_index()
            test = test.reset_index(drop = True)
            self.__test = test
            
            #print values
            print("Ratio's:\nTrain: {}\nTest: {}".format(len(train)/len(data), len(test)/len(data)))
            train_ratios = np.array(train.groupby(self.__column).count().iloc[:,0])/np.array(lengths)
            
            spot = 0
            for i in list(train.groupby(self.__column).count().index):
                print("\nTrain Ratio's:")
                print('{}: {}'.format(i, train_ratios[spot]))
                print('Ratio of {} to train: {}'.format(i, len(index[i])/len(train)))
                print('Cases of {} in train: {}'.format(i, len(index[i])))
                print("\nTest Ratio's:")
                print('Ratio of {} to test: {}'.format(i, (-len(index[i])+lengths[i])/len(test)))
                print('Cases of {} in test: {}\n'.format(i, (-len(index[i])+lengths[i])))
                spot += 1
            
        return train, test
            
        
        

#lengths = list(data.groupby('Class').count().iloc[:,0])
##get index values
#index = {}
#for i in range(0,len(data['Class'].unique())):
#    index[i]= list(data[data['Class']==i].sample(n = int(.8*lengths[i]), 
#    random_state = 229, axis = 0).index)
##make datasets
#train = data_cont.iloc[index[0]+index[1]+index[2],:]
#test = data_cont.drop(index = index[0]+index[1]+index[2])
##print ratios
#print("Ratio's:\nTrain: {}\nTest: {}".format(len(train)/len(data_cont), len(test)/len(data_cont)))
#train_ratios = np.array(train.groupby('Class').count().iloc[:,0])/np.array(lengths)
#print("\nTrain Ratio's:\nLow Wattage: {}\nMed Wattage: {}\nhigh Wattage:{}"\
#      .format(train_ratios[0],train_ratios[1],train_ratios[2]))

            
        