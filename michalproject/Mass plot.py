# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 06:20:41 2017

@author: michal
"""

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


import numpy as np
import matplotlib.pyplot as plt
data = np.array(np.genfromtxt("/home/michal/Desktop/Final Year project/raw data/datasetHs.csv",delimiter=","))


n=[]
nn=1
M=[]
for i in data:
    if 4<5: #PT cut
        dphi=i[2]-i[9]
        deta=i[0]-i[7]
        dr=np.sqrt(dphi**2+deta**2)
        m=dr*i[15]/2
        M.append(m)
        n.append(nn)
        nn=nn+1
    
    

plt.figure(1)
plt.hist(M,40)
plt.show(1)
