from os import system
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

system('clear')


"""print '------------------------------ Extract Data and Fix SubjectIDs--------------------------------'
#### In this file FiberAnalysis/main.py or FiberAnalysis/main_BasedOn2yearOlds.py can be used interchangeably 

system('python ExtractData_FixSubID.py')
"""
print '------------------------------ Cognitive Data ------------------------------'
print 'Running ADHD_BRIEF_BASC.py ...'
#Preproccess data and build the csv
system('python ADHD_BRIEF_BASC.py')

print 'Running Diffrent Labeling Criteria on Brief and Basc and labeling ...'
#Usees Cognitive data at school entery to label datas into 2 categories at-risk and typical
system('python select_labels.py')

print 'Extracts balanced sample labels ...'
system('python sample_labels.py')

print '--------------------------Transfer Cognitive and Brain data------------------'
system('python copyFiles.py')

print '--------------------------- Neonate Brain data preparing --------------------'
system('python ./neo_files/neo_main.py')

print '--------------------------- 1-year Brain data preparing ---------------------'
system('python ./1year_files/1year_main.py')

print '--------------------------- 2-year Brain data preparing ---------------------'
system('python ./2year_files/2year_main.py')

print '--------------------------- Longitudenal Brain data preparing ---------------------'
system('python ./long_files/long_main.py')

print 'This part only can be run if we run LT.m in matlab:'
print '-----------------------Merge neonate-1year-2year uncomplete data after Latent tree data imputation ---------------------'
system('python ./LT_files/concats_neo1year2year.py')

print '---------------------------- Classification ---------------------------------'
system('python Classification.py')






