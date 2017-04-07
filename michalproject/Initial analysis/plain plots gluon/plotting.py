# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 02:41:52 2017

@author: michal
"""
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


import numpy as np
import matplotlib.pyplot as plt


datag = np.array(np.genfromtxt("/home/michal/Desktop/Final Year project/raw data/dataset.csv",delimiter=","))
#dataH = np.genfromtxt("/home/michal/Desktop/Final Year project/raw data/datasetH.csv",delimiter=",")

plt.figure(1)
plt.title(r"$\eta$ of b")
plt.xlabel(r"$\eta$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[0]],20,normed=True)
plt.savefig("etab.png" , dpi=175, bbox_inches="tight")
plt.close(1)

plt.figure(2)
plt.title(r"$p_{T}$ of b")
plt.xlabel(r"$p_{T}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[1]],20,normed=True)
plt.savefig("ptb.png" , dpi=175, bbox_inches="tight")
plt.close(2)

plt.figure(3)
plt.title(r"$\phi$ of b")
plt.xlabel(r"$\phi$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[2]],20,normed=True)
plt.savefig("phib.png" , dpi=175, bbox_inches="tight")
plt.close(3)

plt.figure(4)
plt.title(r"$p_{x}$ of b")
plt.xlabel(r"$p_{x}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[3]],20,normed=True)
plt.savefig("pxb.png" , dpi=175, bbox_inches="tight")
plt.close(4)

plt.figure(5)
plt.title(r"$p_{y}$ of b")
plt.xlabel(r"$p_{y}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[4]],20,normed=True)
plt.savefig("pyb.png" , dpi=175, bbox_inches="tight")
plt.close(5)

plt.figure(6)
plt.title(r"$p_{z}$ of b")
plt.xlabel(r"$p_{z}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[5]],20,normed=True)
plt.savefig("pzb.png" , dpi=175, bbox_inches="tight")
plt.close(6)

plt.figure(7)
plt.title("E of b")
plt.xlabel("E")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[6]],20,normed=True)
plt.savefig("Eb.png" , dpi=175, bbox_inches="tight")
plt.close(7)



plt.figure(8)
plt.title(r"$\eta$ of $\bar{b}$")
plt.xlabel(r"$\eta$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[7]],20,normed=True)
plt.savefig("etabbar.png" , dpi=175, bbox_inches="tight")
plt.close(8)

plt.figure(9)
plt.title(r"$p_{T}$ of $\bar{b}$")
plt.xlabel(r"$p_{T}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[8]],20,normed=True)
plt.savefig("ptbbar.png" , dpi=175, bbox_inches="tight")
plt.close(9)

plt.figure(10)
plt.title(r"$\phi$ of $\bar{b}$")
plt.xlabel(r"$\phi$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[9]],20,normed=True)
plt.savefig("phibbar.png" , dpi=175, bbox_inches="tight")
plt.close(10)

plt.figure(11)
plt.title(r"$p_{x}$ of $\bar{b}$")
plt.xlabel(r"$p_{x}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[10]],20,normed=True)
plt.savefig("pxbbar.png" , dpi=175, bbox_inches="tight")
plt.close(11)

plt.figure(12)
plt.title(r"$p_{y}$ of $\bar{b}$")
plt.xlabel(r"$p_{y}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[11]],20,normed=True)
plt.savefig("pybbar.png" , dpi=175, bbox_inches="tight")
plt.close(12)

plt.figure(13)
plt.title(r"$p_{z}$ of $\bar{b}$")
plt.xlabel(r"$p_{z}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[12]],20,normed=True)
plt.savefig("pzbbar.png" , dpi=175, bbox_inches="tight")
plt.close(13)

plt.figure(14)
plt.title(r"E of $\bar{b}$")
plt.xlabel("E")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[13]],20,normed=True)
plt.savefig("Ebbar.png" , dpi=175, bbox_inches="tight")
plt.close(14)


plt.figure(15)
plt.title(r"$\eta$ of g")
plt.xlabel(r"$\eta$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[14]],20,normed=True)
plt.savefig("etag.png" , dpi=175, bbox_inches="tight")
plt.close(15)

plt.figure(16)
plt.title(r"$p_{T}$ of g")
plt.xlabel(r"$p_{T}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[15]],20,normed=True)
plt.savefig("ptg.png" , dpi=175, bbox_inches="tight")
plt.close(16)

plt.figure(17)
plt.title(r"$\phi$ of g")
plt.xlabel(r"$\phi$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[16]],20,normed=True)
plt.savefig("phig.png" , dpi=175, bbox_inches="tight")
plt.close(17)

plt.figure(18)
plt.title(r"$p_{x}$ of g")
plt.xlabel(r"$p_{x}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[17]],20,normed=True)
plt.savefig("pxg.png" , dpi=175, bbox_inches="tight")
plt.close(18)

plt.figure(19)
plt.title(r"$p_{y}$ of g")
plt.xlabel(r"$p_{y}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[18]],20,normed=True)
plt.savefig("pyg.png" , dpi=175, bbox_inches="tight")
plt.close(19)

plt.figure(20)
plt.title(r"$p_{z}$ of g")
plt.xlabel(r"$p_{z}$")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[19]],20,normed=True)
plt.savefig("pzg.png" , dpi=175, bbox_inches="tight")
plt.close(20)

plt.figure(21)
plt.title("E of g")
plt.xlabel("E")
plt.ylabel("Relative freq.")
plt.hist(datag[:,[20]],20,normed=True)
plt.savefig("Eg.png" , dpi=175, bbox_inches="tight")
plt.close(21)