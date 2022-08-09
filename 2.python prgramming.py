# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:54:31 2019

@author: visu
"""



#Set working directory
import os
os.chdir("C:\\Users\\visu\\Desktop\\R\\python")
os.getcwd()
#Create array
import numpy as np
np.arange(10)
#Create list
lis = ["India", 5, 6, 7]
sample_matrix = np.matrix('1 2 3; 3 4 5')
#transpose matrix
np.transpose(sample_matrix)
# Arithmetic with matrices
sample_matrix*2
sample_matrix/2
#Let us learn how to create data frame
import pandas as pd
df = pd.DataFrame({'age': [1,2,4,9], 
                   'Gender': ["M", "F", "F", "M"],
                   'Income': [5,6,7,3]})
#Rename varaible
df = df.rename(columns = {'Gender':'Gender_v1', 'Income':'Income_v1'})
#Selecting variables from the dataset and making a new dataset
from ggplot import mtcars
df_mtcars = mtcars
df_mtcars_subset = df_mtcars[['hp', 'cyl', 'carb']]
#selecting the rows with condition; use &
df_mtcars_v1 = df_mtcars.loc[(df_mtcars['cyl'] == 6) | (df_mtcars['carb'] == 4)]
#Creating new variable
import numpy as np
df_mtcars['NewVar'] = np.log2(df_mtcars['cyl'])
# Adding rows of two data frame
d1 = pd.DataFrame({'a' : [3,4,5,6], 'b' : [4,5,6,6]}) #Let us create two data frame to merge
d2 = pd.DataFrame({'a' : [5,6,7,8], 'b' : [4,5,7,8]})
​
d3 = d1.append(d2)
#Adding columns 
d3 = pd.concat([d1.reset_index(drop=True), d2], axis=1)
#convert each varaible
df_mtcars['cyl'] = df_mtcars['cyl'].astype(object)
#convert entire dataframe
df_mtcars = df_mtcars.astype(object)
#Convert numeric to categorical/binning
df_mtcars['mpgcat']= np.where(df_mtcars['mpg'] > 20, 'Low', 0)
##Sorting in ascending order
df_mtcars = df_mtcars.sort_values('mpg', ascending = True)
##Sorting in desceding order
df_mtcars = df_mtcars.sort_values('mpg', ascending = False)
#Merge dataframes
df1 = pd.DataFrame({ 'A' : range(1, 6 ,1),
                     'B' : np.random.randint(5, 105 , size = 5),
                     'C' : ['Toaster','Toaster','Ohio', 'Toaster','Ohio'] })
​
df2 = pd.DataFrame({'A' : [2, 4, 5],
                    'Product' : ['Medium', 'Classic', 'Delux']})
#Joins
df_left = df1.merge(df2, on = 'A' ,how = 'left')
df_right = df1.merge(df2, on = 'A' ,how = 'right')
#replace all values in a varaible
x = ("This is a sentence about axis")
x.replace("is", "XY")
​
df_mtcars
​
#############2ndclass############################
Python 3 
File
Edit
View
Insert
Cell
Kernel
Widgets
Help

#Set working directory
import os
os.chdir("G:\Analytics\Edwisor\Version_Revamp\Predictive Analytics using R\ModifiedVersion\Python Code")
#Create vector
Player1 = [16, 9, 13, 5, 2, 17, 14]
#Loops in python
for i in Player1:
    print(i)
german['newcol'] = 0
for i in range(len(german)):
    german['newcol'].loc[i] = german['Amount'].loc[i] + german['Duration'].loc[i]
​
german['newcol_1'] = 0
for i in range(len(german)):
    if(german['ResidenceDuration'].iloc[i] == 4):
        german['newcol_1'].iloc[i] = 1    
#Apply functions in python
import numpy as np
german.iloc[:,0:4].apply(np.sum, axis = 0)

