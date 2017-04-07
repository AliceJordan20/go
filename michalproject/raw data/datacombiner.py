# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:15:41 2017

@author: michal

This script combines all the data files from each run of Pythia. It is important
to rename the runs as simply n with no file extension, with n being intergers from 1
to N , the total number of files to combine.
"""
import numpy as np

#lcmbr listcombiner opens a file, reads each line into a list and appends it to oldlist
def lcmbr(filename,oldlist):
    file = open(str(filename),"r")
    dataset=file.readlines()
    file.close()
    dataset=dataset[143:]
    return (oldlist+dataset)

#initializing variables
data =[] 
data1=[] 
n    =1

#for loop to add all the files to the data list
while n < 21:
    data=lcmbr(n,data)
    n=n+1


#This removes any rows that are not data but artifacts of pythia
for i in data:
    j=np.fromstring(i,sep="\t")
    if len(j) ==21:
        data1.append(j)
    
#converting into a numpy array and saving
data2=np.array(data1)
np.savetxt("dataset.csv",data2,delimiter=",",fmt="%1.3f") 
"""
IMPORTANT

The data is saved in a np.array of dimensions 21xN where N is the amount of data
the columns represent the following
eta pT phi px py pz E
that makes 7 variables for b, bbar and the parent g in order. 
"""