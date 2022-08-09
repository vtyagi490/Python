# -*- coding: utf-8 -*-

#Strings
ll ="this is python"
print(ll)
type(ll)
mstr = " hi how, are you"
mstr[0] 
mstr[-1]
mstr[4]
#To get first word
a =(mstr.split( )[1])
print(a)
# split string with dilimeter
print(mstr.split(","))
type(mstr)
# list
x = [1, 50, 30, 4, 5,8,9,54,33,563,3555]
x
x[0]
x[1]
x[2]
x[-1]
x[-2]
type(x)
x[:3]
x[2:]
k1 = [1, 2, 3]
k2 = [4, 5, 6]
z =k1+k2
print(z)
import numpy as np
z=np.add(k1,k2)
print(z)
# Repeat List N times
X = [1, 2, 3]
Z = X * 3
Z
# Modify / Replace a list item
X = [1, 2, 3,7,5]
X[2]=5
print(X)
# Add / Remove a list item
a = ['AA', 'BB', 'CC']
a
a.append('DD')
print(a)
a.remove('BB')
a
a.insert(0,"bb")
a
help
print(a)
x= 'kk'
# Sort list
k = [124, 225, 305, 246, 259]
k.sort()
print(k)
    # Tuple
K  = (1,2,3,7,8)
type(K)
K[0]=7
State =("kk","ll","dd")
type(State)
# Tuple cannot be altered
X = (1, 2, 3)
X[2]=5
X
#  Dictionary
teams = {'Dave' : 'team A',
         'Tim' : 'team B',
         'Babita' : 'team C',
         'Sam' : 'team B',
         'Ravi' : 'team C'
        }
type(teams)
print(teams)
# Find Values
teams['Sam']
# Delete an item
del teams['Ravi']
print(teams)
#Add an item
teams['Deep'] = 'team B'
print(teams)
# Sets
##Sets are unordered collections of simple objects.
X = set(['A', 'B', 'C'])
type(X)
# Does 'A' exist in set X
'A' in X 
'D' in X
# add 'D' in set X
X.add('D')
#remove 'C' from set X
X.remove('C')
#create a copy of set X
Y = X.copy()
# common items in both sets X and Y
Y & X
