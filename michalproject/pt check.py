# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 05:33:26 2017

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

#phi b is 2
#pt b is 1
#phi bbar is 9
#pt bbar is 8
#phi H is 16
#pt H is 15

    
n=[]
nn=1
netPT=[]
netptp=[]
for i in data:
    netPT.append( np.sqrt(i[4]**2+i[3]**2)-i[1])
    netptp.append( 100*(np.sqrt(i[4]**2+i[3]**2)-i[1])/i[1] )
    n.append(nn)
    nn=nn+1
    



plt.figure(1)
plt.title(r"$P_{T}$ Disagreement")
plt.ylabel("Magnitude")
plt.scatter(n[::50],netPT[::50],marker="x")
plt.show(1)

plt.figure(2)
plt.title(r"$P_{T}$ Disagreement Histogram")
plt.ylabel("freq,")
plt.xlabel("Magnitude")
plt.hist(netPT,40)
plt.show(2)

plt.figure(3)
plt.title(r"$P_{T}$ percentage")
plt.ylabel("freq,")
plt.xlabel("Magnitude")
plt.hist(netptp,40)
plt.show(3)