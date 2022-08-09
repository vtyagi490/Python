# -*- coding: utf-8 -*-
import pandas as pd
data = pd.read_csv("F:\\R\\data\\sampledata.csv")
data.shape
data.shape[0]
data.shape[1]
# Get Variable Names
data.columns
list(data)
ad=data.head()
ad
# Knowing the Variable types
type(data)
data.dtypes
data['State'].dtypes
# Changing the data types
data.Y2008 = data.Y2008.astype(float)
data.dtypes
data.to_csv("F:\\R\\data\\sampledata1.csv")
# Extract Unique Values
data.Index.unique()
# The nunique( ) shows the number of unique values
data.Index.nunique()
data.State.nunique()
# Generate Cross Tab
pd.crosstab(data.Index,data.State)
# frequency distribution
data.Index.value_counts(ascending = True)
# to get ramdom sample
dat1=data.sample(n = 5)
dat1.shape
data.sample(frac = 0.1)
# select few columns
d1=data.loc[:,["Index","State","Y2008"]]
d1.shape
list(d1)
#Selecting consecutive columns
data1 =data.loc[:,"Index":"Y2008"]
list(data1)
dat6=data.iloc[:,0:5] 
data2 =data[["Index","State","Y2008"]]
data2.shape
# Renaming the variables
dataset = pd.DataFrame({"A" : ["John","Mary","Julia","Kenny","Henry"], "B" : ["Libra","Capricorn","Aries","Scorpio","Aquarius"]})
dataset
#Renaming all the variables.
dataset.columns = ['Names','Zodiac Signs'] 
dataset
#Renaming only some of the variables.
dataset.rename(columns = {"Names":"Cust_Name"},inplace = True)
dataset
# Removing the columns and rows
d2  =data.drop('Index',axis = 1)
list(d2)
# #Alternatively
data.drop("Index",axis = "columns")
d4=data.drop(['Index','State'],axis = 1)
list(d4)
d5 =data.drop(0,axis = 0)
data.shape
d5.shape
data.drop(50,axis = "index")
d6 =data.drop([0,1,2,3],axis = 0)
d6.shape
#Sorting the data
dat1 =data.sort_values("State",ascending = False)
print(dat1.head())
data.sort_values("State",ascending = False,inplace = True)
dat2 =data.Y2006.sort_values() 
print(dat2.head())
dat3 =data.sort_values(["Index","Y2002"]) 
dat3.head()
# Create new variables
data["difference"] = data.Y2008-data.Y2009
list(data)
#Alternatively
data["difference2"] = data.eval("Y2008 - Y2009")
data.head()
data["ratio"]= data.Y2008/data.Y2009
data.head()
list(data)
data1 = data.assign(ratio1 = (data.Y2008 / data.Y2009))
data1.head()
# Descriptive Statistics
#for numeric variables
data.Y2009.describe()
data.describe()
##Only for strings / objects
data.describe(include = ['object'])
data.Y2008.mean()
data.Y2008.median()
data.Y2008.min()
data.loc[:,["Y2002","Y2008"]].max()
# Groupby function
data.groupby("Index").Y2008.min()
data.groupby("Index")["Y2008","Y2010"].max()

a =data.groupby("Index").Y2002.agg(["count","min","max","mean","var"])
b=data.groupby("Index")["Y2002","Y2003"].agg(["count","min","max","mean"])

data.groupby("Index").agg({"Y2002": ["min","max"],"Y2003" : "mean"})
# Filtering
dat7=data[data.Index == "A"]
#Alternatively
d =data.loc[data.Index == "A",:]
data.loc[data.Index == "A","State"]
data.loc[data.Index == "A",:].State

# filter the rows with Index as "A" and income for 2002 > 1500000"
dat8=data.loc[(data.Index == "A") & (data.Y2002 > 1500000),:]
# filter the rows with index either "A" or "W
dat9 =data.loc[(data.Index == "A") | (data.Index == "W"),:]

#Alternatively.
data.loc[data.Index.isin(["A","W"]),:]

data.query('Y2002>1700000 & Y2003 > 1500000')
# Dealing with missing values
import pandas as pd
import numpy as np
mydata = {'Crop': ['Rice', 'Wheat', 'Barley', 'Maize'],
        'Yield': [1010, 1025.2, 1404.2, 1251.7],
        'cost' : [102, np.nan, 20, 68]}
crops = pd.DataFrame(mydata)
crops
type(crops)
#same as is.na in R
crops.isnull() 
# #opposite of previous command.
crops.notnull()
# #No. of missing values.
crops.isnull().sum()
# #shows the rows with NAs.
crops[crops.cost.isnull()]
 #shows the rows with NAs in crops.Crop
crops[crops.cost.isnull()].Crop
#shows the rows without NAs in crops.Crop
crops[crops.cost.notnull()].Crop 

a=crops.dropna(how = "any")

crops.dropna(how = "all").shape 

crops.dropna(subset = ['Yield',"cost"],how = 'any').shape
crops.dropna(subset = ['Yield',"cost"],how = 'all').shape
# Replacing the missing values by "UNKNOWN"
crops['cost'].fillna(value = "UNKNOWN",inplace = True)
crops
 # Dealing with duplicates
data = pd.DataFrame({"Items" : ["TV","Washing Machine","Mobile","TV","TV","Washing Machine"], "Price" : [10000,50000,20000,10000,10000,40000]})
data

data.loc[data.duplicated(),:]
data.loc[data.duplicated(keep = "first"),:]
#last entries are not there,indices have changed.
data.loc[data.duplicated(keep = "last"),:]
#all the duplicates, including unique are shown.
data.loc[data.duplicated(keep = False),:] 
data.drop_duplicates(keep = "first")
data.drop_duplicates(keep = "last")
data.drop_duplicates(keep = False)
# if and else
students = pd.DataFrame({'Names': ['John','Mary','Henry','Augustus','Kenny'],
                         'Zodiac Signs': ['Aquarius','Libra','Gemini','Pisces','Virgo']})

def name(row):
    if row["Names"] in ["John","Henry"]:
        return("yes")
    else:
        return("no")
 
students.flag.describe()
students['flag'] = students.apply(name, axis=1)
students
import numpy as np

students['flag1'] = np.where(students['Names'].isin(['John','Henry']), 'yes', 'no')
students
# Multiple Conditions : If Else-if Else
def mname(row):
    if row["Names"] == "John" and row["Zodiac Signs"] == "Aquarius" :
        return "yellow"
    elif row["Names"] == "Mary" and row["Zodiac Signs"] == "Libra" :
        return "blue"
    elif row["Zodiac Signs"] == "Pisces" :
        return "blue"
    else:
        return "black"

students['color'] = students.apply(mname, axis=1)
students

# Merging or joining on the basis of common variable.

students = pd.DataFrame({'Names': ['John','Mary','Henry','Maria'],
'Zodiac Signs': ['Aquarius','Libra','Gemini','Capricorn']})

students2 = pd.DataFrame({'Names': ['John','Mary','Henry','Augustus','Kenny'],
'Marks' : [50,81,98,25,35]})
    
    # intersections result
result = pd.merge(students, students2, on='Names')
result
#unions result
result1 = pd.merge(students, students2, on='Names',how = "outer") 
result1
# left
result2 = pd.merge(students, students2, on='Names',how = "left")
result2 
# right
result3 = pd.merge(students, students2, on='Names',how = "right", indicator =True)
result3  
 