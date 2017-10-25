import pandas as pd


def union(a,b):
	return list(set(a)|set(b))

def intersect(a,b):
	return list(set(a) & set(b))

def subtract(a,b):
	return list(set(a)-set(b))



def Generate_BrainCogData(BrainCog,BrainData,writeTo):
	CT_cols = [col for col in BrainCog.columns if 'ct_' in col.strip().lower()]
	_CT_cols = [col for col in BrainCog.columns if '_ct' in col.strip().lower()]
	SA_cols = [col for col in BrainCog.columns if 'sa_' in col.strip().lower()]
	_SA_cols = [col for col in BrainCog.columns if '_sa' in col.strip().lower()]

	CT = CT_cols + _CT_cols
	SA = SA_cols + _SA_cols

	ColsWithoutCT_SA = subtract(BrainCog.columns,CT)
	ColsWithoutCT_SA = subtract(ColsWithoutCT_SA,SA)
	BrainSubsetFeats = BrainCog[ColsWithoutCT_SA]

	BrainSubsetFeats = BrainSubsetFeats.dropna()
	cols = BrainSubsetFeats.columns.tolist()
	new_orderCols=['subjectID']+subtract(cols,['subjectID','Label'])+['Label']
	BrainSubsetFeats = BrainSubsetFeats[new_orderCols]
	BrainSubsetFeats.to_csv('./neo_files/neo_BrainSubsetFeats%s.csv' %writeTo,index=False)

	DCT_cols = [col for col in BrainData.columns if 'ct_' in col.strip().lower()]
	_DCT_cols = [col for col in BrainData.columns if '_ct' in col.strip().lower()]
	DSA_cols = [col for col in BrainData.columns if 'sa_' in col.strip().lower()]
	_DSA_cols = [col for col in BrainData.columns if '_sa' in col.strip().lower()]

	DCT = DCT_cols + _DCT_cols
	DSA = DSA_cols + _DSA_cols


	ColsWithoutDCT_SA = subtract(BrainData.columns,DCT)
	ColsWithoutDCT_SA = subtract(ColsWithoutDCT_SA,DSA)
	BrainSubsetFeats = BrainData[ColsWithoutDCT_SA]

	BrainSubsetFeats = BrainSubsetFeats.dropna()
	cols = BrainSubsetFeats.columns.tolist()
	new_orderCols=['subjectID']+subtract(cols,['subjectID'])
	BrainSubsetFeats = BrainSubsetFeats[new_orderCols]
	BrainSubsetFeats.to_csv('./neo_files/neo_NoLabelBrainSubsetFeats%s.csv' %writeTo,index=False)

BrainCog = pd.read_csv('./neo_files/neo_BrainUncomp.csv')
BrainData = pd.read_csv('./neo_files/neo_UnBrainData.csv') 
Generate_BrainCogData(BrainCog,BrainData,'')

BrainCog = pd.read_csv('./neo_files/neo_BrainUncomp_balanced.csv')
Generate_BrainCogData(BrainCog,BrainData,'_balanced')
