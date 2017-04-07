# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:13:05 2017

@author: michal
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC


data_train = np.loadtxt( "training.csv", delimiter=",")

np.random.seed(9001)

r=np.random.rand(data_train.shape[0])

#cent is the cutoff, so 0.9 will use 90% of th data for training, last 10% for validation
cent=0.9

"""
Y= truth
X= Data
I= Index
"""

Y_TRAIN = data_train[:,13][r<cent]
X_TRAIN = data_train[:,0:12][r<cent]

Y_VALID = data_train[:,13][r>=cent]
X_VALID = data_train[:,0:12][r>=cent]

gbc = GBC(n_estimators=50 , max_depth = 5, min_samples_leaf=200,max_features=10,verbose=1)
gbc.fit(X_TRAIN,Y_TRAIN)

prediction_TRAIN= gbc.predict_proba(X_TRAIN)[:,1]
prediction_VALID= gbc.predict_proba(X_VALID)[:,1]

pcut= np.percentile(prediction_TRAIN,85)

Y2_TRAIN = prediction_TRAIN > pcut
Y2_VALID = prediction_VALID > pcut



data_test = np.loadtxt( "test.csv", delimiter=",")

X_TEST = data_test[:,0:12]
I_TEST = range(0,X_TEST.shape[0])

prediction_TEST= gbc.predict_proba(X_TEST)[:,1]
Label_TEST     = list(prediction_TEST > pcut)
prediction_TEST= list(prediction_TEST)

results=[]

for i in range(len(I_TEST)):
    results.append([int(I_TEST[i]), prediction_TEST[i], "s"*(Label_TEST[i]==1.0)+"b"*(Label_TEST==0.0)])
    
results=sorted(results,key=lambda a_entry:a_entry[1])

for i in range(len(results)):
    results[i][1]=i+1
results=sorted(results,key=lambda a_entry:a_entry[0])

fcsv= open("Prediction_output.csv","w")
fcsv.write("Number,Rank Order,Classification\n")
for line in results:
    theline = str(line[0])+","+str(line[1])+","+line[2]+"\n"
    fcsv.write(theline)
fcsv.close()
















from matplotlib import pyplot as plt
 
Classifier_training_S = gbc.predict_proba(X_TRAIN[Y_TRAIN>0.5])[:,1].ravel()
Classifier_training_B = gbc.predict_proba(X_TRAIN[Y_TRAIN<0.5])[:,1].ravel()
Classifier_testing_A = gbc.predict_proba(X_TEST)[:,1].ravel()
  
c_max = max([Classifier_training_S.max(),Classifier_training_B.max(),Classifier_testing_A.max()])
c_min = min([Classifier_training_S.min(),Classifier_training_B.min(),Classifier_testing_A.min()])
  
# Get histograms of the classifiers
Histo_training_S = np.histogram(Classifier_training_S,bins=50,range=(c_min,c_max))
Histo_training_B = np.histogram(Classifier_training_B,bins=50,range=(c_min,c_max))
Histo_testing_A = np.histogram(Classifier_testing_A,bins=50,range=(c_min,c_max))
  
# Lets get the min/max of the Histograms
AllHistos= [Histo_training_S,Histo_training_B]
h_max = max([histo[0].max() for histo in AllHistos])*1.2
# h_min = max([histo[0].min() for histo in AllHistos])
h_min = 1.0
  
# Get the histogram properties (binning, widths, centers)
bin_edges = Histo_training_S[1]
bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
bin_widths = (bin_edges[1:] - bin_edges[:-1])
  
# To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
ErrorBar_testing_A = np.sqrt(Histo_testing_A[0])
# ErrorBar_testing_B = np.sqrt(Histo_testing_B[0])
  
# Draw objects
ax1 = plt.subplot(111)
  
# Draw solid histograms for the training data
ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)
ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],bottom=Histo_training_B[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
 
ff = (1.0*(sum(Histo_training_S[0])+sum(Histo_training_B[0])))/(1.0*sum(Histo_testing_A[0]))
 
# # Draw error-bar histograms for the testing data
ax1.errorbar(bin_centers, ff*Histo_testing_A[0], yerr=ff*ErrorBar_testing_A, xerr=None, ecolor='black',c='black',fmt='.',label='Test (reweighted)')
# ax1.errorbar(bin_centers, Histo_testing_B[0], yerr=ErrorBar_testing_B, xerr=None, ecolor='red',c='red',fmt='o',label='B (Test)')
  
# Make a colorful backdrop to show the clasification regions in red and blue
ax1.axvspan(pcut, c_max, color='blue',alpha=0.08)
ax1.axvspan(c_min,pcut, color='red',alpha=0.08)
  
# Adjust the axis boundaries (just cosmetic)
ax1.axis([c_min, c_max, h_min, h_max])
  
# Make labels and title
plt.title("Higgs Signal-Background Separation")
plt.xlabel("Probability Output (Gradient Boosting)")
plt.ylabel("Counts/Bin")
 
# Make legend with smalll font
legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
for alabel in legend.get_texts():
            alabel.set_fontsize('small')
  
# Save the result to png
plt.savefig("Sklearn_gbc.png")