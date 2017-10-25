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

def Generate_BrainUncomp(writeTo,Brain_cog,B):

	Brain_cog=pd.merge(Brain_cog,B,on='subjectID')
	print 'Merge of Labels and 2year_subcortical.csv:',Brain_cog.shape

	print sum(Brain_cog['Label'])
	print sum(Brain_cog['Label'])*100.0/(Brain_cog.shape[0])

	Brain_cog=replace_space(Brain_cog)
	
	Brain_cog.to_csv(writeTo,index=False)


all_files = []
Join_files=['./2year_files/2year_subcortical.csv','LABELS.csv']
rest_files = ['./2year_files/2year_GM90Region.csv','./2year_files/2year_Vent_Vol.csv','./2year_files/2year_CT.csv','./2year_files/2year_SA_Middle.csv','./2year_files/2year_CT_Average.csv','./2year_files/2year_SA_MiddleTotal.csv',
'./2year_files/2year_WMProbMapBoxParc.csv','./2year_files/2year_CSFProbMapBoxParc.csv','./2year_files/2year_GMProbMapBoxParc.csv','./2year_files/2year_ICV_Info.csv']
all_files = Join_files+rest_files

for brain_info_file in all_files:
	
	Brain_Info = pd.read_csv(brain_info_file)
	Brain_Info['subjectID'] = [c.strip() for c in Brain_Info['subjectID']]
	print brain_info_file, Brain_Info.shape	
	Brain_Info.to_csv(brain_info_file,index=False)	

Brain_cog = pd.read_csv('./2year_files/2year_subcortical.csv')

print 'Merge of Labels and 2year_subcortical.csv:',Brain_cog.shape

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

	Brain_cog=pd.merge(Brain_Info,Brain_cog,how='outer',on='subjectID')
	print Brain_cog.shape

B= pd.read_csv('LABELS.csv')
Generate_BrainUncomp('./2year_files/2year_BrainUncomp.csv',Brain_cog,B)

B = pd.read_csv('LABELS_balanced.csv')
Generate_BrainUncomp('./2year_files/2year_BrainUncomp_balanced.csv',Brain_cog,B)

