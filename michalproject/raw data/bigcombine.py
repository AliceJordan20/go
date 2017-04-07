#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:16:31 2017

@author: michal
"""
"""
This program takes both gluon and higgs datasets and
generates an overall set to be used for machine learning
whilst labelling them
"""
import numpy as np

gluons=np.loadtxt("datasets.csv",delimiter=",")
higgs =np.loadtxt("datasetHss.csv",delimiter=",")
newhiggs=np.ones((len(higgs),len(higgs[1])+1))
newhiggs[:,:-1] = higgs
newgluons=np.zeros((len(gluons),len(gluons[1])+1))
newgluons[:,:-1] = gluons
finalmatrix=np.concatenate((newgluons,newhiggs))
np.random.shuffle(finalmatrix)
np.savetxt("test.csv",finalmatrix,delimiter=",",fmt="%1.3f")
