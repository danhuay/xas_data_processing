# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from xas_bin import xasData
import os

os.chdir("H:\\Google Drive\\PhD\\Research\\Data\\NSLS-II\\2017.3.302162_PEC\\PEC_SCAN2")
files = os.listdir()
for filename in files:
    rawdata = xasData(edge=4966, filename=filename)
    rawdata.saveBinned()
    print(filename+' binned!')