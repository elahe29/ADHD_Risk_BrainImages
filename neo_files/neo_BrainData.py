import pandas as pd

def replace_space(df):
	
	cols = df.columns 
	#cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x) 
	cols = [col.strip().replace(' ','_') for col in cols]
	cols = [col.strip().replace('(mm^2)','mm2') for col in cols]
	df.columns = cols
	return df

def change_col_name(df,prefix):
	cols=df.columns
	cols = cols.map(lambda x: prefix+x if x!='subjectID' else x) 
	#cols = [prefix+col for col in cols if col!='subjectID' else col]
	#print cols
	df.columns = cols
	return df

all_files = []
Join_files=['./neo_files/neo_HippoAmy.csv']
rest_files = ['./neo_files/neo_Vent_Vol.csv','./neo_files/neo_CT.csv','./neo_files/neo_SA_Middle.csv','./neo_files/neo_CT_Average.csv','./neo_files/neo_SA_MiddleTotal.csv','./neo_files/neo_WMProbMapBoxParc.csv','./neo_files/neo_EMProbMapBoxParc.csv','./neo_files/neo_CSFProbMapBoxParc.csv','./neo_files/neo_GMProbMapBoxParc.csv','./neo_files/neo_ICV_Info.csv','./neo_files/neo_TalaCaudate.csv']
all_files = Join_files+rest_files

for brain_info_file in all_files:
	
	Brain_Info = pd.read_csv(brain_info_file)
	Brain_Info['subjectID'] = [c.strip() for c in Brain_Info['subjectID']]
	print brain_info_file, Brain_Info.shape	
	Brain_Info.to_csv(brain_info_file,index=False)	

BrainData=pd.read_csv('./neo_files/neo_HippoAmy.csv')
UnBrainData=pd.read_csv('./neo_files/neo_HippoAmy.csv')
print BrainData.shape

for brain_info_file in rest_files:
	print brain_info_file
	Brain_Info = pd.read_csv(brain_info_file)
	if brain_info_file=='./neo_files/neo_CT.csv':
		Brain_Info=change_col_name(Brain_Info,'CT_')	
	if brain_info_file=='./neo_files/neo_SA_Middle.csv':
		Brain_Info=change_col_name(Brain_Info,'SA_')
	if brain_info_file=='./neo_files/neo_WMProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'WM_')
	if brain_info_file=='./neo_files/neo_EMProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'EM_')
	if brain_info_file=='./neo_files/neo_GMProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'GM_')
	if brain_info_file=='./neo_files/neo_CSFProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'CSF_')

	#Brain_Info['subjectID'] = [c.strip() for c in Brain_Info['subjectID']]
	BrainData=pd.merge(Brain_Info,BrainData,how='inner',on='subjectID')
	UnBrainData=pd.merge(Brain_Info,UnBrainData,how='outer',on='subjectID')
	print BrainData.shape

BrainData=replace_space(BrainData)
UnBrainData=replace_space(UnBrainData)
BrainData.to_csv('./neo_files/neo_BrainData.csv',index=False)
UnBrainData.to_csv('./neo_files/neo_UnBrainData.csv',index=False)
