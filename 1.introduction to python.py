# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:54:18 2019

@author: visu
"""



#Set working directory
import os
os.chdir("C:/Users/visu/Desktop/R")

#Check current working directory
os.getcwd()
#Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib as mlt
#Load csv data in python
df_csv = pd.read_csv("German.csv", sep = ',')
#load data from csv
df_csv = pd.read_excel("German.xlsx")
#Getting the column names of the dataset
df_csv.columns
#Getting the structure of the dataset
type(df_csv)
#Getting the number of variables and obervation in the datasets
df_csv.shape
#Getting first 10 rows of the dataset
df_csv.head(3)
#Getting the last 10 rows of the dataset
df_csv.tail(3)
#delete dataframe
del df_csv
#Getting first 10 rows of the dataset
df_csv.iloc[:10,:4]
#select first 4 columns
df_csv.iloc[:,:4]
#Unique values in a column
df_csv['InstallmentRatePercentage'].unique()
#Count of unique values in a column
df_csv['InstallmentRatePercentage'].nunique()
#Distribution of unique values in a column
df_csv['InstallmentRatePercentage'].value_counts()
#Summary of a varaible 
df_csv['InstallmentRatePercentage'].describe()
#Assignment operators
df = 25 #storing numeric value
df1 = "I love India"      #storing characters
#Calculate statistical quantities
df_csv['InstallmentRatePercentage'].mean()
df_csv['InstallmentRatePercentage'].median()
df_csv['InstallmentRatePercentage'].var()
df_csv['InstallmentRatePercentage'].std()
#Add all the values of column
df_csv['InstallmentRatePercentage'].sum()
# Writing a csv (output)
df_csv.to_csv("df_csv_practice.csv", index = False)
# Writing a xlsx file
df_csv.to_excel("df_excel_practice.xlsx", index = False)
​
