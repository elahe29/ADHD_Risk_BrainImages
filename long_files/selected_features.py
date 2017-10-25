import pandas as pd,csv
from os import system
#system('clear')

def change_col_name(df,prefix):
	cols = df.columns
	cols = cols.map(lambda x: prefix+x if (x!='subjectID') else x) 
	#print cols
	df.columns = cols
	return df

def union(a,b):
	return list(set(a)|set(b))
def intersect(a,b):
	return list(set(a) & set(b))
def subtract(a,b):
	return list(set(a)-set(b))

Comp2year = pd.read_csv('./2year_files/2year_BrainUncomp.csv')
Comp1year = pd.read_csv('./1year_files/1year_BrainUncomp.csv')
CompNeo = pd.read_csv('./neo_files/neo_BrainUncomp.csv')

Comp2year.columns = [x.upper() for x in Comp2year.columns]
Comp1year.columns = [x.upper() for x in Comp1year.columns]
CompNeo.columns = [x.upper() for x in CompNeo.columns]

print "Similar cols in 1 year_2year",len(intersect(Comp1year.columns,Comp2year.columns))
print "UnSimilar cols in 1 year_2year",len(union(Comp1year.columns,Comp2year.columns))-len(intersect(Comp1year.columns,Comp2year.columns))

Neo1yearUni = union(Comp1year.columns,CompNeo.columns)
Neo1yearInt = intersect(Comp1year.columns,CompNeo.columns)

print "\nSimilar cols in 1_year Neo", len(Neo1yearInt)
print "UnSimilar cols in 1_year Neo",len(Neo1yearUni)-len(Neo1yearInt)

print "\nCols in Neos, Cols in 1year_olds",len(CompNeo.columns),len(Comp1year.columns)
print "Neonate columns which doesn't exist in 1year",len(CompNeo.columns)-len(Neo1yearInt)
print "1year_olds columns which doesn't exist in Neonate",(len(Comp1year.columns)-len(Neo1yearInt))

print "\nCols in Neonate not is 1 year", subtract(CompNeo.columns,Comp1year.columns)
print "\nCols in Neonate not is 1 year", subtract(Comp1year.columns,CompNeo.columns)

Neo1year2yearInt = intersect(Comp2year,Neo1yearInt)
print "Similat Cols in all time points", len(Neo1year2yearInt)



