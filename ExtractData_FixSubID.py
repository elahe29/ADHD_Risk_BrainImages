import os

root='/work/efarah/Data/'
pythonRunList= ['DemogData/main.py','CognitiveData/main.py','SubCortical/main.py','SurfArea-CortT/main.py','Volumetric/main.py','TBSS/main.py','FiberAnalysis/neo_ReadFiber.py']


print '------------------------------ Extract Demographic Data--------------------------------'
currDir=('%sDemogData' %root)
print currDir

os.chdir(currDir)
os.system('python ./main.py')

print '------------------------------ Extract Cognitive Data--------------------------------'
currDir=('%sCognitiveData' %root)
print currDir

os.chdir(currDir)
os.system('python ./main.py')

print '------------------------------ Extract SubCortical Data--------------------------------'
currDir=('%sSubCortical' %root)
print currDir

os.chdir(currDir)
os.system('python ./main.py')

print '------------------------------ Extract Surface Area-Corttical Thickness Data--------------------------------'
currDir=('%sSurfArea-CortT' %root)
print currDir

os.chdir(currDir)
os.system('python ./main.py')

print '------------------------------ Extract Volumetric Data--------------------------------'
currDir=('%sVolumetric' %root)
print currDir

os.chdir(currDir)
os.system('python ./main.py')

print '------------------------------ Extract TBSS(Enigma) Data--------------------------------'
currDir=('%sTBSS' %root)
print currDir

os.chdir(currDir)
os.system('python ./main.py')

print '------------------------------ Extract FiberTract Data--------------------------------'
currDir=('%sFiberAnalysis' %root)
print currDir

os.chdir(currDir)
#os.system('python ./concat_DF2Atlas.py')
os.system('python ./main_BasedOn2yearOlds.py')
#os.system('python ./main.py')

