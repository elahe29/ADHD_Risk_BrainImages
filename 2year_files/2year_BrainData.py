import pandas as pd

def replace_space(df):
	
	cols = df.columns  
	cols = [col.strip().replace(' ','_') for col in cols]
	cols = [col.strip().replace('(mm^2)','mm2') for col in cols]
	df.columns = cols
	return df

def change_col_name(df,prefix):
	cols=df.columns
	cols = cols.map(lambda x: prefix+x if x!='subjectID' else x) 
	#print cols
	df.columns = cols
	return df


all_files = []
Join_files=['./2year_files/2year_subcortical.csv']
rest_files = ['./2year_files/2year_GM90Region.csv','./2year_files/2year_Vent_Vol.csv','./2year_files/2year_CT.csv','./2year_files/2year_SA_Middle.csv','./2year_files/2year_CT_Average.csv','./2year_files/2year_SA_MiddleTotal.csv',
'./2year_files/2year_WMProbMapBoxParc.csv','./2year_files/2year_CSFProbMapBoxParc.csv','./2year_files/2year_GMProbMapBoxParc.csv','./2year_files/2year_ICV_Info.csv']
all_files = Join_files+rest_files

for brain_info_file in all_files:
	
	Brain_Info = pd.read_csv(brain_info_file)
	Brain_Info['subjectID'] = [c.strip() for c in Brain_Info['subjectID']]
	print brain_info_file, Brain_Info.shape	
	Brain_Info.to_csv(brain_info_file,index=False)	

BrainData=pd.read_csv('./2year_files/2year_subcortical.csv')
UnBrainData=pd.read_csv('./2year_files/2year_subcortical.csv')
print 'Shape of 2year_subcortical.csv:',BrainData.shape

for brain_info_file in rest_files:
	print brain_info_file
	Brain_Info = pd.read_csv(brain_info_file)
	if brain_info_file=='./2year_files/2year_CT.csv':
		Brain_Info=change_col_name(Brain_Info,'CT_')	
	if brain_info_file=='./2year_files/2year_SA_Middle.csv':
		Brain_Info=change_col_name(Brain_Info,'SA_')
	if brain_info_file=='./2year_files/2year_WMProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'WM_')
	if brain_info_file=='./2year_files/2year_GMProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'GM_')
	if brain_info_file=='./2year_files/2year_CSFProbMapBoxParc.csv':
		Brain_Info=change_col_name(Brain_Info,'CSF_')
	if brain_info_file=='./2year_files/2year_GM90Region.csv':
		Brain_Info=change_col_name(Brain_Info,'90RG_')
	UnBrainData=pd.merge(Brain_Info,UnBrainData,how='outer',on='subjectID')
	BrainData=pd.merge(Brain_Info,BrainData,how='inner',on='subjectID')
print BrainData.shape

BrainData=replace_space(BrainData)
UnBrainData=replace_space(UnBrainData)

BrainData.to_csv('./2year_files/2year_BrainData.csv',index=False)
UnBrainData.to_csv('./2year_files/2year_UnBrainData.csv',index=False)
