import pandas as pd
import numpy as np
from os import system
import csv


def find_corrupted_digits(Data):
	print 'Fix Corrupted Data ...'
	for col in Data.columns:
		Data[col] = Data[col].astype(str)	
	count_dot = Data[Data.columns].applymap(lambda x: str.count(x, '.'))
	for col in Data.columns:
		for inx in range(len(count_dot[col])):
			if(count_dot.iloc[inx][col]>1):
				
				cell_value = Data.iloc[inx][col]
				cell_value = cell_value[0:cell_value.find('.',cell_value.find('.')+1)]	
				Data.iloc[inx][col] = cell_value
	for col in Data.columns:
		if col !='subjectID':		
			Data[col] = Data[col].astype(float)	
	return Data			

def change_col_name(df,prefix):

	cols = df.columns	
	cols = cols.map(lambda x: prefix+x if (x!='subjectID') else x)
	#print cols	
	df.columns = cols
	return df

def FirstNotNull(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]

def union(a,b):
	return list(set(a)|set(b))

def intersect(a,b):
	return list(set(a) & set(b))

def subtract(a,b):
	return list(set(a)-set(b))

def shared_features_func():

	Comp2year = pd.read_csv('./2year_files/2year_BrainUncomp.csv')
	Comp1year = pd.read_csv('./1year_files/1year_BrainUncomp.csv')
	CompNeo = pd.read_csv('./neo_files/neo_BrainUncomp.csv')

	print "Similar cols in 1 year_2year",len(intersect(Comp1year.columns,Comp2year.columns))
	print "UnSimilar cols in 1 year_2year",len(union(Comp1year.columns,Comp2year.columns))-len(intersect(Comp1year.columns,Comp2year.columns))

	Neo1yearUni = union(Comp1year.columns,CompNeo.columns)
	Neo1yearInt = intersect(Comp1year.columns,CompNeo.columns)

	print "\nSimilar cols in 1_year Neo", len(Neo1yearInt)
	print "UnSimilar cols in 1_year Neo",len(Neo1yearUni)-len(Neo1yearInt)

	print "\nCols in Neos, Cols in 1year_olds",len(CompNeo.columns),len(Comp1year.columns)
	print "Neonate columns which doesn't exist in 1year",len(CompNeo.columns)-len(Neo1yearInt)
	print "1year_olds columns which doesn't exist in Neonate",(len(Comp1year.columns)-len(Neo1yearInt))

	#print "\nCols in Neonate not is 1 year", subtract(CompNeo.columns,Comp1year.columns)
	#print "\nCols in Neonate not is 1 year", subtract(Comp1year.columns,CompNeo.columns)

	Neo1year2yearInt = intersect(Comp2year,Neo1yearInt)

	return Neo1year2yearInt
	

def CT_SA_features():
	

	BrainCog1year = pd.read_csv('./1year_files/1year_BrainUncomp.csv')
	BrainCog2year = pd.read_csv('./2year_files/2year_BrainUncomp.csv')
	BrainCogNeo = pd.read_csv('./neo_files/neo_BrainUncomp.csv')
	
	allCols	= union(union(BrainCogNeo.columns , BrainCog1year.columns) , BrainCog2year.columns)
		
	CT_cols = [col for col in allCols if 'ct_' in col.strip().lower()]
	_CT_cols = [col for col in allCols if '_ct' in col.strip().lower()]
	SA_cols = [col for col in allCols if 'sa_' in col.strip().lower()]
	_SA_cols = [col for col in allCols if '_sa' in col.strip().lower()]

	CT = CT_cols + _CT_cols
	SA = SA_cols + _SA_cols
	
	CTSA= CT+SA

	return CTSA

def Delta_alltime(shared_feats,demogData,cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,longData,labled):
	longData = pd.merge(longData,demogData,how='left',on='subjectID')
	
	if labled:
		ds0 = longData[cols_neo]
		ds1 = longData[cols_1year]
		ds2 = longData[cols_2year]

		long_delta = pd.DataFrame(columns=['subjectID']+subtract(cols_1yearNeo,['subjectID','Label'])+subtract(cols_2year1year,['subjectID','Label'])+['Label'])
	else:
		ds0 = longData[subtract(cols_neo,['Label'])]
		ds1 = longData[subtract(cols_1year,['Label'])]
		ds2 = longData[subtract(cols_2year,['Label'])]
		cols_1yearNeo = subtract(cols_1yearNeo,['Label'])
		cols_2year1year = subtract(cols_2year1year,['Label'])

		long_delta = pd.DataFrame(columns=['subjectID']+subtract(cols_1yearNeo,['subjectID'])+subtract(cols_2year1year,['subjectID']))

	ds0=find_corrupted_digits(ds0)
	ds1=find_corrupted_digits(ds1)
	ds2=find_corrupted_digits(ds2)

	for feature in subtract(shared_feats,['subjectID','Label']):

		ScanDate_2year = pd.to_datetime(longData['2year_ScanDate'])		
		ScanDate_1year = pd.to_datetime(longData['1year_ScanDate'])
		ScanDate_neo = pd.to_datetime(longData['neo_ScanDate'])
		
		# some coding shoudl be add here that if one of the Scan_dates are missing the norm for that part should be 1

		norm_1yearNeo = 365.0/(ScanDate_1year-ScanDate_neo).astype('timedelta64[D]')
		norm_2year1year = 365.0/(ScanDate_2year-ScanDate_1year).astype('timedelta64[D]')

		long_delta['1yearNeo_'+feature]=(ds1['1year_'+feature].astype(float)-ds0['neo_'+feature].astype(float))*norm_1yearNeo
		long_delta['2year1year_'+feature]=(ds2['2year_'+feature].astype(float)-ds1['1year_'+feature].astype(float))*norm_2year1year

	if labled:	
		long_delta[['subjectID','Label']]=ds1[['subjectID','Label']]
		cols = long_delta.columns.tolist()
		new_orderCols=['subjectID']+subtract(cols,['subjectID','Label'])+['Label']
		long_delta = long_delta[new_orderCols]
	else:
		long_delta[['subjectID']]=ds1[['subjectID']]
		cols = long_delta.columns.tolist()
		new_orderCols=['subjectID']+subtract(cols,['subjectID'])
		long_delta = long_delta[new_orderCols]
	
	return long_delta

def Delta_twotimes(shared_feats,demogData,cols_time1,cols_time2,cols_time2time1,Data_time2time1,labled,duration):

	Data_time2time1 = pd.merge(Data_time2time1,demogData,how='left',on='subjectID')
	
	if labled:
		ds0 = Data_time2time1[cols_time1]
		ds1 = Data_time2time1[cols_time2]

		df_time2time1 = pd.DataFrame(columns=['subjectID']+subtract(cols_time2time1,['subjectID','Label'])+['Label'])
	else:
		ds0 = Data_time2time1[subtract(cols_time1,['Label'])]
		ds1 = Data_time2time1[subtract(cols_time2,['Label'])]

		cols_time2time1 = subtract(cols_time2time1,['Label'])
		df_time2time1 = pd.DataFrame(columns=['subjectID']+subtract(cols_time2time1,['subjectID']))

	ds0=find_corrupted_digits(ds0)
	ds1=find_corrupted_digits(ds1)

	for feature in subtract(shared_feats,['subjectID','Label']):

		ScanDate_time2 = pd.to_datetime(Data_time2time1['1year_ScanDate'])
		ScanDate_time1 = pd.to_datetime(Data_time2time1['neo_ScanDate'])
		
		# some coding shoudl be add here that if one of the Scan_dates are missing the norm for that part should be 1

		norm_time2time1 = 365.0/(ScanDate_time2-ScanDate_time1).astype('timedelta64[D]')
		if duration =='1yearNeo':
			df_time2time1['1yearNeo_'+feature]=(ds1['1year_'+feature].astype(float)-ds0['neo_'+feature].astype(float))*norm_time2time1
		else:
			df_time2time1['2year1year_'+feature]=(ds1['2year_'+feature].astype(float)-ds0['1year_'+feature].astype(float))*norm_time2time1

	if labled:	
		df_time2time1[['subjectID','Label']]=ds1[['subjectID','Label']]
		cols = df_time2time1.columns.tolist()
		new_orderCols=['subjectID']+subtract(cols,['subjectID','Label'])+['Label']
		df_time2time1 = df_time2time1[new_orderCols]
	else:
		df_time2time1[['subjectID']]=ds1[['subjectID']]
		cols = df_time2time1.columns.tolist()
		new_orderCols=['subjectID']+subtract(cols,['subjectID'])
		df_time2time1 = df_time2time1[new_orderCols]

	return df_time2time1


	
def Generate_longDelta(shared_feats,cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,Data_1yearNeo,Data_2year1year,longData,labled):

	demogData = pd.read_csv('ConteTwinDemog.csv')
	demogData = demogData[['subjectID','neo_ScanDate','1year_ScanDate','2year_ScanDate']]
	
	long_delta = Delta_alltime(shared_feats,demogData,cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,longData,labled)
	long_delta_1yearNeo = Delta_twotimes(shared_feats,demogData,cols_neo,cols_1year,cols_1yearNeo,Data_1yearNeo,labled,'1yearNeo')
	long_delta_2year1year = Delta_twotimes(shared_feats,demogData,cols_1year,cols_2year,cols_2year1year,Data_2year1year,labled,'2year1year')
	
	return long_delta,long_delta_1yearNeo,long_delta_2year1year

def Generate_FiberData(Fibertype,labled,writeTo):
	
	Fiber2year = pd.read_csv('./2year_files/2year_'+Fibertype+'%s.csv' %writeTo)
	Fiber2year = change_col_name(Fiber2year,'2year_')
	Fiber1year = pd.read_csv('./1year_files/1year_'+Fibertype+'%s.csv' %writeTo)
	Fiber1year = change_col_name(Fiber1year,'1year_')
	FiberNeo = pd.read_csv('./neo_files/neo_'+Fibertype+'%s.csv' %writeTo)
	FiberNeo = change_col_name(FiberNeo,'neo_')

	outer_FiberLong12=pd.merge(Fiber2year,Fiber1year,how='outer',on='subjectID')
	outer_FiberLong=pd.merge(outer_FiberLong12,FiberNeo,how='outer',on='subjectID')
	if labled:
		outer_FiberLong['Label'] = pd.concat([outer_FiberLong['1year_Label'],outer_FiberLong['2year_Label'],outer_FiberLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
		outer_FiberLong.drop([col for col in ['1year_Label','2year_Label','neo_Label'] if col in outer_FiberLong],axis=1,inplace=True)
	outer_FiberLong.to_csv('./long_files/longUncomp_'+Fibertype+'%s.csv' %writeTo,index=False)

	inner_FiberLong12=pd.merge(Fiber2year,Fiber1year,how='inner',on='subjectID')
	inner_FiberLong=pd.merge(inner_FiberLong12,FiberNeo,how='inner',on='subjectID')
	if labled:
		inner_FiberLong['Label'] = pd.concat([inner_FiberLong['1year_Label'],inner_FiberLong['2year_Label'],inner_FiberLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
		inner_FiberLong.drop([col for col in ['1year_Label','2year_Label','neo_Label'] if col in inner_FiberLong],axis=1,inplace=True)
	inner_FiberLong.to_csv('./long_files/longcomp_'+Fibertype+'%s.csv' %writeTo,index=False)

	print "\nShapes of "+Fibertype+" Data: 2year :",Fiber2year.shape,' 1year :',Fiber1year.shape,' Neonate :',FiberNeo.shape,'complete long: 1year2year :',inner_FiberLong12.shape,' complete long: Neo1year2year :',inner_FiberLong.shape
	print "Shapes of  "+Fibertype+" Data: 2year :",Fiber2year.shape,' 1year :',Fiber1year.shape,' Neonate :',FiberNeo.shape,'uncomplete long: 1year2year :',outer_FiberLong12.shape,'uncomplete long: Neo1year2year :',outer_FiberLong.shape


def main_script(writeTo):

################# Load data and prepare data
	BrainSubset2year = pd.read_csv('./2year_files/2year_BrainSubsetFeats%s.csv' %writeTo)
	BrainSubset1year = pd.read_csv('./1year_files/1year_BrainSubsetFeats%s.csv' %writeTo)
	BrainSubsetNeo = pd.read_csv('./neo_files/neo_BrainSubsetFeats%s.csv' %writeTo)
	BrainSubset2year = change_col_name(BrainSubset2year,'2year_')	
	BrainSubset1year = change_col_name(BrainSubset1year,'1year_')
	BrainSubsetNeo = change_col_name(BrainSubsetNeo,'neo_')

	Brain_cog2year = pd.read_csv('./2year_files/2year_Brain_cog%s.csv' %writeTo)
	Brain_cog1year = pd.read_csv('./1year_files/1year_Brain_cog%s.csv' %writeTo)
	Brain_cogNeo = pd.read_csv('./neo_files/neo_Brain_cog%s.csv' %writeTo)
	Brain_cog2year = change_col_name(Brain_cog2year,'2year_')
	Brain_cog1year = change_col_name(Brain_cog1year,'1year_')
	Brain_cogNeo = change_col_name(Brain_cogNeo,'neo_')

	Comp2year = pd.read_csv('./2year_files/2year_BrainUncomp%s.csv' %writeTo)
	Comp1year = pd.read_csv('./1year_files/1year_BrainUncomp%s.csv' %writeTo)
	CompNeo = pd.read_csv('./neo_files/neo_BrainUncomp%s.csv' %writeTo)
	Comp2year = change_col_name(Comp2year,'2year_')
	Comp1year = change_col_name(Comp1year,'1year_')
	CompNeo = change_col_name(CompNeo,'neo_')

	BrainData2year = pd.read_csv('./2year_files/2year_BrainData.csv')
	BrainData1year = pd.read_csv('./1year_files/1year_BrainData.csv')
	BrainDataNeo = pd.read_csv('./neo_files/neo_BrainData.csv')
	BrainData2year = change_col_name(BrainData2year,'2year_')
	BrainData1year = change_col_name(BrainData1year,'1year_')
	BrainDataNeo = change_col_name(BrainDataNeo,'neo_')

	NoLabelBrainSubset2year = pd.read_csv('./2year_files/2year_NoLabelBrainSubsetFeats.csv')
	NoLabelBrainSubset1year = pd.read_csv('./1year_files/1year_NoLabelBrainSubsetFeats.csv')
	NoLabelBrainSubsetNeo = pd.read_csv('./neo_files/neo_NoLabelBrainSubsetFeats.csv')
	NoLabelBrainSubset2year = change_col_name(NoLabelBrainSubset2year,'2year_')
	NoLabelBrainSubset1year = change_col_name(NoLabelBrainSubset1year,'1year_')
	NoLabelBrainSubsetNeo = change_col_name(NoLabelBrainSubsetNeo,'neo_')


	print "1 year olds cols:",len(Comp1year.columns)
	print "2 year olds cols:",len(Comp2year.columns)
	print "Neonates cols:",len(CompNeo.columns)

	Neo1yearcols = [col for col in CompNeo.columns if col in Comp1year.columns]
	NeoNot1yearcols=[col for col in CompNeo.columns if col not in Comp1year.columns]
	OneyearNotNeocols=[col for col in Comp1year.columns if col not in CompNeo.columns]

	print "\nSimilar cols in 1 year and Neo:",len(Neo1yearcols)
	print '\nNeonate Columns not in 1year old.....................','non Similar cols in 1 year and Neo:',len(NeoNot1yearcols)
	#print NeoNot1yearcols
	print '\n1year old Columns not in Neonate.....................','non Similar cols in 1 year and Neo:',len(OneyearNotNeocols)
	#print OneyearNotNeocols



	outer_CompLong12 = pd.merge(Comp2year,Comp1year,how='outer',on='subjectID')
	outer_CompLong = pd.merge(outer_CompLong12,CompNeo,how='outer',on='subjectID')

	outer_CompLong['Label'] = pd.concat([outer_CompLong['1year_Label'],outer_CompLong['2year_Label'],outer_CompLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)

	outer_CompLong.drop(['1year_Label','2year_Label','neo_Label'],axis=1,inplace=True)
	outer_CompLong.to_csv('./long_files/longUncomp_BrainUncomp%s.csv' %writeTo,index=False)

	inner_CompLong12 = pd.merge(Comp2year,Comp1year,how='inner',on='subjectID')
	inner_CompLong = pd.merge(inner_CompLong12,CompNeo,how='inner',on='subjectID')

	inner_CompLong['Label'] = pd.concat([inner_CompLong['1year_Label'],inner_CompLong['2year_Label'],inner_CompLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)

	inner_CompLong.drop(['1year_Label','2year_Label','neo_Label'],axis=1,inplace=True)
	inner_CompLong.to_csv('./long_files/longcomp_BrainUncomp%s.csv' %writeTo,index=False)

	print "\nShapes of Uncomplete Data: 2year :",Comp2year.shape,' 1year :',Comp1year.shape,' Neonate :',CompNeo.shape,'complete long:1year2year :',inner_CompLong12.shape,'complete long: Neo1year2year :',inner_CompLong.shape
	print "Shapes of Uncomplete Data: 2year :",Comp2year.shape,' 1year :',Comp1year.shape,' Neonate :',CompNeo.shape,'uncomplete long: 1year2year :',outer_CompLong12.shape,'uncomplete long: Neo1year2year :',outer_CompLong.shape

	outer_BrainDataLong12=pd.merge(BrainData2year,BrainData1year,how='outer',on='subjectID')
	outer_BrainDataLong=pd.merge(outer_BrainDataLong12,BrainDataNeo,how='outer',on='subjectID')
	outer_BrainDataLong.to_csv('./long_files/longUncomp_BrainData.csv',index=False)

	inner_BrainDataLong12=pd.merge(BrainData2year,BrainData1year,how='inner',on='subjectID')
	inner_BrainDataLong=pd.merge(inner_BrainDataLong12,BrainDataNeo,how='inner',on='subjectID')
	inner_BrainDataLong.to_csv('./long_files/longcomp_BrainData.csv',index=False)

	print "\nShapes of complete UnlabledData: 2year :",BrainData2year.shape,' 1year :',BrainData1year.shape,' Neonate :',BrainDataNeo.shape,'complete long: 1year2year :',inner_BrainDataLong12.shape,'complete long: Neo1year2year :',inner_BrainDataLong.shape
	print "Shapes of complete UnlabledData: 2year :",BrainData2year.shape,' 1year :',BrainData1year.shape,' Neonate :',BrainDataNeo.shape,'uncomplete long: 1year2year :',outer_BrainDataLong12.shape,' uncomplete long: Neo1year2year :',outer_BrainDataLong.shape

	outer_Brain_cogLong12=pd.merge(Brain_cog2year,Brain_cog1year,how='outer',on='subjectID')
	outer_Brain_cogLong=pd.merge(outer_Brain_cogLong12,Brain_cogNeo,how='outer',on='subjectID')
	outer_Brain_cogLong['Label'] = pd.concat([outer_Brain_cogLong['1year_Label'],outer_Brain_cogLong['2year_Label'],outer_Brain_cogLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
	outer_Brain_cogLong.drop([col for col in ['1year_Label','2year_Label','neo_Label'] if col in outer_Brain_cogLong],axis=1,inplace=True)
	outer_Brain_cogLong.to_csv('./long_files/longUncomp_Brain_cog%s.csv' %writeTo,index=False)

	inner_Brain_cogLong12=pd.merge(Brain_cog2year,Brain_cog1year,how='inner',on='subjectID')
	inner_Brain_cogLong=pd.merge(inner_Brain_cogLong12,Brain_cogNeo,how='inner',on='subjectID')
	inner_Brain_cogLong['Label'] = pd.concat([inner_Brain_cogLong['1year_Label'],inner_Brain_cogLong['2year_Label'],inner_Brain_cogLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
	inner_Brain_cogLong.drop([col for col in ['1year_Label','2year_Label','neo_Label'] if col in inner_Brain_cogLong],axis=1,inplace=True)
	inner_Brain_cogLong.to_csv('./long_files/longcomp_Brain_cog%s.csv' %writeTo,index=False)

	print "\nShapes of complete LabledData: 2year :",Brain_cog2year.shape,' 1year :',Brain_cog1year.shape,' Neonate :',Brain_cogNeo.shape,'complete long: 1year2year :',inner_Brain_cogLong12.shape,' complete long: Neo1year2year :',inner_Brain_cogLong.shape
	print "Shapes of complete LabledData: 2year :",Brain_cog2year.shape,' 1year :',Brain_cog1year.shape,' Neonate :',Brain_cogNeo.shape,'uncomplete long: 1year2year :',outer_Brain_cogLong12.shape,'uncomplete long: Neo1year2year :',outer_Brain_cogLong.shape


	inner_BrainSubsetLong12=pd.merge(BrainSubset2year,BrainSubset1year,how='inner',on='subjectID')
	inner_BrainSubsetLong=pd.merge(inner_BrainSubsetLong12,BrainSubsetNeo,how='inner',on='subjectID')
	inner_BrainSubsetLong['Label'] = pd.concat([inner_BrainSubsetLong['1year_Label'],inner_BrainSubsetLong['2year_Label'],inner_BrainSubsetLong['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
	inner_BrainSubsetLong.drop([col for col in ['1year_Label','2year_Label','neo_Label'] if col in inner_BrainSubsetLong],axis=1,inplace=True)
	inner_BrainSubsetLong.to_csv('./long_files/longcomp_BrainSubset%s.csv' %writeTo,index=False)

	print "\nShapes of complete BrainSubset: 2year :",BrainSubset2year.shape,' 1year :',BrainSubset1year.shape,' Neonate :',BrainSubsetNeo.shape,'complete long: 1year2year :',inner_BrainSubsetLong12.shape,'complete long: Neo1year2year :',inner_BrainSubsetLong.shape

	
	inner_NoLabelBrainSubsetLong12=pd.merge(NoLabelBrainSubset2year,NoLabelBrainSubset1year,how='inner',on='subjectID')
	inner_NoLabelBrainSubsetLong=pd.merge(inner_NoLabelBrainSubsetLong12,NoLabelBrainSubsetNeo,how='inner',on='subjectID')
	inner_NoLabelBrainSubsetLong.to_csv('./long_files/longcomp_NoLabelBrainSubset.csv',index=False)

	print "\nShapes of complete NoLabelBrainSubset: 2year :",NoLabelBrainSubset2year.shape,' 1year :',NoLabelBrainSubset1year.shape,' Neonate :',NoLabelBrainSubsetNeo.shape,'complete long: 1year2year :',inner_NoLabelBrainSubsetLong12.shape,'complete long: Neo1year2year :',inner_NoLabelBrainSubsetLong.shape

	shared_features = shared_features_func()


	cols_neo = map(lambda x: 'neo_'+x if (x!='subjectID' and x!='Label') else x,shared_features)
	cols_1year = map(lambda x: '1year_'+x if (x!='subjectID' and x!='Label') else x,shared_features)
	cols_2year = map(lambda x: '2year_'+x if (x!='subjectID' and x!='Label') else x,shared_features) 

	cols_1yearNeo = map(lambda x: '1yearNeo_' + x if (x!='subjectID' and x!='Label') else x,shared_features)
	cols_2year1year = map(lambda x: '2year1year_' + x if (x!='subjectID' and x!='Label') else x,shared_features)

	Brain_cog_2year1year=pd.merge(Brain_cog2year,Brain_cog1year,how='inner',on='subjectID')
	Brain_cog_2year1year['Label'] = pd.concat([Brain_cog_2year1year['1year_Label'],Brain_cog_2year1year['2year_Label']],axis=1).apply(FirstNotNull, axis=1)
	Brain_cog_2year1year.drop([col for col in ['1year_Label','2year_Label'] if col in Brain_cog_2year1year],axis=1,inplace=True)

	Brain_cog_1yearNeo=pd.merge(Brain_cog1year,Brain_cogNeo,how='inner',on='subjectID')
	Brain_cog_1yearNeo['Label'] = pd.concat([Brain_cog_1yearNeo['1year_Label'],Brain_cog_1yearNeo['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
	Brain_cog_1yearNeo.drop([col for col in ['1year_Label','neo_Label'] if col in Brain_cog_1yearNeo],axis=1,inplace=True)


	BrainData_2year1year=pd.merge(BrainData2year,BrainData1year,how='inner',on='subjectID')
	BrainData_1yearNeo=pd.merge(BrainData1year,BrainDataNeo,how='inner',on='subjectID')


	[long_delta_Brain_cog,long_delta_1yearNeo_Brain_cog,long_delta_2year1year_Brain_cog] = Generate_longDelta(shared_features,cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,Brain_cog_1yearNeo,Brain_cog_2year1year,inner_Brain_cogLong,True)
	long_delta_Brain_cog.to_csv('./long_files/longdelta_Brain_cog%s.csv' %writeTo,index=False)
	long_delta_1yearNeo_Brain_cog.to_csv('./long_files/longdelta_1yearNeo_Brain_cog%s.csv' %writeTo,index=False)
	long_delta_2year1year_Brain_cog.to_csv('./long_files/longdelta_2year1year_Brain_cog%s.csv' %writeTo,index=False)

	[long_delta_BrainData,long_delta_1yearNeo_BrainData,long_delta_2year1year_BrainData] = Generate_longDelta(shared_features,cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,BrainData_1yearNeo,BrainData_2year1year,inner_BrainDataLong,False)
	long_delta_BrainData.to_csv('./long_files/longdelta_BrainData.csv',index=False)
	long_delta_1yearNeo_BrainData.to_csv('./long_files/longdelta_1yearNeo_BrainData.csv',index=False)
	long_delta_2year1year_BrainData.to_csv('./long_files/longdelta_2year1year_BrainData.csv',index=False)

	CTSA = CT_SA_features()
	cols_neo = map(lambda x: 'neo_'+x if (x!='subjectID' and x!='Label') else x,subtract(shared_features,CTSA))
	cols_1year = map(lambda x: '1year_'+x if (x!='subjectID' and x!='Label') else x,subtract(shared_features,CTSA))
	cols_2year = map(lambda x: '2year_'+x if (x!='subjectID' and x!='Label') else x,subtract(shared_features,CTSA)) 

	cols_1yearNeo = map(lambda x: '1yearNeo_' + x if (x!='subjectID' and x!='Label') else x,subtract(shared_features,CTSA))
	cols_2year1year = map(lambda x: '2year1year_' + x if (x!='subjectID' and x!='Label') else x,subtract(shared_features,CTSA))


	BrainSubset_2year1year=pd.merge(BrainSubset2year,BrainSubset1year,how='inner',on='subjectID')
	BrainSubset_2year1year['Label'] = pd.concat([BrainSubset_2year1year['1year_Label'],BrainSubset_2year1year['2year_Label']],axis=1).apply(FirstNotNull, axis=1)
	BrainSubset_2year1year.drop([col for col in ['1year_Label','2year_Label'] if col in BrainSubset_2year1year],axis=1,inplace=True)

	BrainSubset_1yearNeo=pd.merge(BrainSubset1year,BrainSubsetNeo,how='inner',on='subjectID')
	BrainSubset_1yearNeo['Label'] = pd.concat([BrainSubset_1yearNeo['1year_Label'],BrainSubset_1yearNeo['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
	BrainSubset_1yearNeo.drop([col for col in ['1year_Label','neo_Label'] if col in BrainSubset_1yearNeo],axis=1,inplace=True)


	NoLabelBrainSubset_2year1year=pd.merge(NoLabelBrainSubset2year,NoLabelBrainSubset1year,how='inner',on='subjectID')
	NoLabelBrainSubset_1yearNeo=pd.merge(NoLabelBrainSubset1year,NoLabelBrainSubsetNeo,how='inner',on='subjectID')


	[long_delta_BrainSubset,long_delta_1yearNeo_BrainSubset,long_delta_2year1year_BrainSubset] = Generate_longDelta(subtract(shared_features,CTSA),cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,BrainSubset_1yearNeo,BrainSubset_2year1year,inner_BrainSubsetLong,True)

	long_delta_BrainSubset.to_csv('./long_files/longdelta_BrainSub%s.csv' %writeTo,index=False)
	long_delta_1yearNeo_BrainSubset.to_csv('./long_files/longdelta_1yearNeo_BrainSub%s.csv' %writeTo,index=False)
	long_delta_2year1year_BrainSubset.to_csv('./long_files/longdelta_2year1year_BrainSub%s.csv' %writeTo,index=False)

	[long_delta_BrainDataSub,long_delta_1yearNeo_BrainDataSub,long_delta_2year1year_BrainDataSub] = Generate_longDelta(subtract(shared_features,CTSA),cols_neo,cols_1year,cols_2year,cols_1yearNeo,cols_2year1year,NoLabelBrainSubset_1yearNeo,NoLabelBrainSubset_2year1year,inner_NoLabelBrainSubsetLong,False)

	long_delta_BrainDataSub.to_csv('./long_files/longdelta_BrainDataSub.csv',index=False)
	long_delta_1yearNeo_BrainDataSub.to_csv('./long_files/longdelta_1yearNeo_BrainDataSub.csv',index=False)
	long_delta_2year1year_BrainDataSub.to_csv('./long_files/longdelta_2year1year_BrainDataSub.csv',index=False)





main_script('')
main_script('_balanced')

Generate_FiberData('Fiber',True,'')
Generate_FiberData('FiberEnigma',True,'')
Generate_FiberData('Fiber',True,'_balanced')
Generate_FiberData('FiberEnigma',True,'_balanced')
Generate_FiberData('UnFiber',False,'')
Generate_FiberData('UnFiberEnigma',False,'')
