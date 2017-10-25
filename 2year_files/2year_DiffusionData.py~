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
	cols = cols.map(lambda x: prefix+str(x) if x!='subjectID' else str(x)) 
	#cols = [prefix+col for col in cols if col!='subjectID' else col]
	#print cols
	df.columns = cols
	return df

def Generate_DiffusionData(writeTo,LabelCSV):
	all_files = []
	Join_files=['./2year_files/2year_df_fa.csv','LABELS.csv']

	rest_files = ['./2year_files/2year_df_ad.csv','./2year_files/2year_df_rd.csv','./2year_files/2year_df_md.csv']
	all_files = Join_files+rest_files

	for dwi_file in all_files:
	
		Brain_Info = pd.read_csv(dwi_file)
		Brain_Info.columns = [c.strip() for c in Brain_Info.columns.tolist()]
		print dwi_file, Brain_Info.shape	
		Brain_Info.to_csv(dwi_file,index=False)	

	Brain_Info = pd.read_csv('./2year_files/2year_df_fa.csv')
	Brain_Info=change_col_name(Brain_Info,'FA_')

	B = pd.read_csv(LabelCSV)

	Brain_cog=pd.merge(Brain_Info,B,on='subjectID')
	print Brain_cog.shape

	for dwi_file in rest_files:
		print dwi_file
		Brain_Info = pd.read_csv(dwi_file)

		if dwi_file=='./2year_files/2year_df_ad.csv':
			Brain_Info=change_col_name(Brain_Info,'AD_')
		if dwi_file=='./2year_files/2year_df_rd.csv':
			Brain_Info=change_col_name(Brain_Info,'RD_')
		if dwi_file=='./2year_files/2year_df_md.csv':
			Brain_Info=change_col_name(Brain_Info,'MD_')
	
	
		Brain_cog = pd.merge(Brain_Info,Brain_cog,on='subjectID')
		print "Fiber:" , Brain_cog.shape

	print sum(Brain_cog['Label'])
	print sum(Brain_cog['Label'])*100.0/(Brain_cog.shape[0])

	Brain_cog=replace_space(Brain_cog)
	Brain_cog.to_csv('./2year_files/2year_Fiber%s.csv' %writeTo,index=False)

	Enigma = pd.read_csv('./2year_files/2year_Enigma.csv')
	FibEnigma = pd.merge(Enigma,Brain_cog,on='subjectID')
	print "Enigma and Fiber:" , FibEnigma.shape
	print sum(FibEnigma['Label'])
	print sum(FibEnigma['Label'])*100.0/(Brain_cog.shape[0])
	FibEnigma = replace_space(FibEnigma)
	FibEnigma.to_csv('./2year_files/2year_FiberEnigma%s.csv' %writeTo,index=False)

Generate_DiffusionData('','LABELS.csv')
Generate_DiffusionData('_balanced','LABELS_balanced.csv')

