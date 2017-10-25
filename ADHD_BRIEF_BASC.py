import pandas as pd

BRIEF = pd.read_csv('BRIEF.csv')
BASC = pd.read_csv('PR_BASC.csv')

print BRIEF.shape
BASC_SUB_ID = BASC['subjectID']
BRIEF = BRIEF.loc[BRIEF['subjectID'].isin(BASC_SUB_ID)]
#print BRIEF.shape

print BASC.shape
BRIEF_SUB_ID = BRIEF['subjectID']
BASC = BASC.loc[BASC['subjectID'].isin(BRIEF_SUB_ID)]
#print BASC.shape

#print BRIEF.columns
BRIEF_ADHD = BRIEF[['subjectID','BRIEF6yr_INH_T','BRIEF6yr_MNT_T','BRIEF6yr_GEC_T','BRIEF6yr_BRI_T','BRIEF6yr_WM_T','BRIEF6yr_MI_T']]
#print BRIEF_ADHD.shape
#print BASC.columns

BASC_ADHD = BASC[['subjectID','PR_BASC6yr_HYP_T','PR_BASC6yr_ExProbCOMP_T','PR_BASC6yr_BSI_COMP_T','PR_BASC6yr_AttProb_T']]
#print BASC_ADHD.shape

BRIEF_BASC_ADHD = pd.merge(BRIEF_ADHD,BASC_ADHD,on='subjectID')
print BRIEF_BASC_ADHD.shape

BRIEF_ADHD.to_csv('BRIEF_ADHD.csv',index=False)
BASC_ADHD.to_csv('BASC_ADHD.csv',index=False)
BRIEF_BASC_ADHD.to_csv('BRIEF_BASC_ADHD.csv',index=False)

