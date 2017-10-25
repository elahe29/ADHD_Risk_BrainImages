from os import system

print 'Running Unlabled Brain data ...'
system('python ./neo_files/neo_BrainData.py')

print 'Running BrainCognitive.py ...'
system('python ./neo_files/neo_BrainCognitive.py')

print 'Running Brain including Null ...'
system('python ./neo_files/neo_BrainUncomp.py')

print 'Selecting subset of Brain features ...'
system('python ./neo_files/neo_BrainSubsetFeaturs.py')

print 'Running Labled DiffusionData.py ...'
system('python ./neo_files/neo_DiffusionData.py')

print 'Running Unlabled DiffusionData.py ...'
system('python ./neo_files/neo_UnDiffusionData.py')


