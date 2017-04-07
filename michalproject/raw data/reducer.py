# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 06:01:07 2017

@author: michal
"""

import numpy as np
data = np.array(np.genfromtxt("/home/michal/Desktop/Final Year project/raw data/datasetH.csv",delimiter=","))
data=data[2::10]
np.savetxt("datasetHss.csv",data,delimiter=",",fmt="%1.3f") 