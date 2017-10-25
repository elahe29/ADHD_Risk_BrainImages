from os import system
import os

def copy(src,dst):
	cmd='cp '+src+' '+dst
	print 'Copy File ' + src + ' to' + dst
	system(cmd)
root='/work/efarah/Data'
files_list= ['CognitiveData','SubCortical','SurfArea-CortT','Volumetric']
def create_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
copyFilestoRoot = ['/DemogData/ConteTwinDemog.csv','/CognitiveData/BRIEF.csv','/CognitiveData/PR_BASC.csv']

copyFilestoNeo = ['/SubCortical/neo_HippoAmy.csv','/SubCortical/neo_TalaCaudate.csv','/SurfArea-CortT/neo_CT.csv','/SurfArea-CortT/neo_CT_Average.csv','/SurfArea-CortT/neo_SA_Middle.csv','/SurfArea-CortT/neo_SA_MiddleTotal.csv','/Volumetric/neo_CSFProbMapBoxParc.csv','/Volumetric/neo_EMProbMapBoxParc.csv','/Volumetric/neo_GMProbMapBoxParc.csv','/Volumetric/neo_WMProbMapBoxParc.csv','/Volumetric/neo_ICV_Info.csv','/Volumetric/neo_ICV_Info.csv','/Volumetric/neo_Vent_Vol.csv','/TBSS/neo_Enigma.csv','/FiberAnalysis/neo_df_ad.csv','/FiberAnalysis/neo_df_rd.csv','/FiberAnalysis/neo_df_fa.csv','/FiberAnalysis/neo_df_md.csv']

copyFilesto1year = ['/SubCortical/1year_subcortical.csv','/SurfArea-CortT/1year_CT.csv','/SurfArea-CortT/1year_CT_Average.csv','/SurfArea-CortT/1year_SA_Middle.csv','/SurfArea-CortT/1year_SA_MiddleTotal.csv','/Volumetric/1year_CSFProbMapBoxParc.csv','/Volumetric/1year_GMProbMapBoxParc.csv','/Volumetric/1year_WMProbMapBoxParc.csv','/Volumetric/1year_ICV_Info.csv','/Volumetric/1year_ICV_Info.csv','/Volumetric/1year_GM90Region.csv','/Volumetric/1year_Vent_Vol.csv','/TBSS/1year_Enigma.csv','/FiberAnalysis/1year_df_ad.csv','/FiberAnalysis/1year_df_rd.csv','/FiberAnalysis/1year_df_fa.csv','/FiberAnalysis/1year_df_md.csv']

copyFilesto2year = ['/SubCortical/2year_subcortical.csv','/SurfArea-CortT/2year_CT.csv','/SurfArea-CortT/2year_CT_Average.csv','/SurfArea-CortT/2year_SA_Middle.csv','/SurfArea-CortT/2year_SA_MiddleTotal.csv','/Volumetric/2year_CSFProbMapBoxParc.csv','/Volumetric/2year_GMProbMapBoxParc.csv','/Volumetric/2year_WMProbMapBoxParc.csv','/Volumetric/2year_ICV_Info.csv','/Volumetric/2year_ICV_Info.csv','/Volumetric/2year_GM90Region.csv','/Volumetric/2year_Vent_Vol.csv','/TBSS/2year_Enigma.csv','/FiberAnalysis/2year_df_ad.csv','/FiberAnalysis/2year_df_rd.csv','/FiberAnalysis/2year_df_fa.csv','/FiberAnalysis/2year_df_md.csv']

for file_ in copyFilestoRoot:
	source=root + file_
	destination=root + '/ADHD-Filter/'
	copy(source,destination)

print '---------------------------create Neonate Folder ----------------------'
create_dir('./neo_files')
for file_ in copyFilestoNeo:
	src=root + file_
	dst=root + '/ADHD-Filter/neo_files'
	copy(src,dst)
print '---------------------------Create 1year olds Folder -------------------'
create_dir('./1year_files')
for file_ in copyFilesto1year:
	src=root + file_
	dst=root + '/ADHD-Filter/1year_files'
	copy(src,dst)
print '---------------------------Create 1year olds Folder -------------------'
create_dir('./2year_files')
for file_ in copyFilesto2year:
	src=root + file_
	dst=root + '/ADHD-Filter/2year_files'
	copy(src,dst)
