import pandas as pd
import csv

def change_col_name(df,prefix):
	cols=df.columns
	cols = cols.map(lambda x: prefix+x if (x!='subjectID') else x) 
	#print cols
	df.columns = cols
	return df


def FirstNotNull(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]

UnCompNeo = pd.read_csv('./neo_files/neo_BrainUncomp.csv')
UnCompNeo = change_col_name(UnCompNeo,'neo_')

UnComp1year = pd.read_csv('./1year_files/1year_BrainUncomp.csv')
LTComp1year = pd.read_csv('./1year_files/1year_BrainLTcomp_noHeader.csv')
LTCompShape = LTComp1year.shape
newLTCop1year=pd.DataFrame(columns=UnComp1year.columns)
#newLTCop1year = change_col_name(newLTCop1year,'1year_')
#UnComp1year = change_col_name(UnComp1year,'1year_')
newLTCop1year['subjectID'] = UnComp1year['subjectID'][0:LTCompShape[0]]
cols = UnComp1year.columns.tolist()
cols.remove('subjectID')
newLTCop1year[cols]=LTComp1year
newLTCop1year.to_csv('./1year_files/1year_BrainLTcomp.csv',index=False)
newLTCop1year = change_col_name(newLTCop1year,'1year_')

UnComp2year = pd.read_csv('./2year_files/2year_BrainUncomp.csv')
LTComp2year = pd.read_csv('./2year_files/2year_BrainLTcomp_noHeader.csv')
LTCompShape = LTComp2year.shape
newLTCop2year=pd.DataFrame(columns=UnComp2year.columns)
#newLTCop2year = change_col_name(newLTCop2year,'2year_')
#UnComp2year = change_col_name(UnComp2year,'2year_')
newLTCop2year['subjectID'] = UnComp2year['subjectID'][0:LTCompShape[0]]
cols = UnComp2year.columns.tolist()
cols.remove('subjectID')
newLTCop2year[cols]=LTComp2year
newLTCop2year.to_csv('./2year_files/2year_BrainLTcomp.csv',index=False)
newLTCop2year = change_col_name(newLTCop2year,'2year_')

LTComp1year2year=pd.merge(newLTCop1year,newLTCop2year,how='inner',on='subjectID')
LTCompneo1year2year=pd.merge(LTComp1year2year,UnCompNeo,how='inner',on='subjectID')

LTCompneo1year2year['Label'] = pd.concat([LTCompneo1year2year['1year_Label'],LTCompneo1year2year['2year_Label'],LTCompneo1year2year['neo_Label']],axis=1).apply(FirstNotNull, axis=1)
LTCompneo1year2year.drop(['1year_Label','2year_Label','neo_Label'],axis=1,inplace=True)

print newLTCop1year.shape,newLTCop2year.shape,LTComp1year2year.shape,LTCompneo1year2year.shape

LTCompneo1year2year.to_csv('./long_files/longcomp_BrainLTcomp_concat.csv',index=False)

