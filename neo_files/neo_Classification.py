from sklearn import preprocessing,svm,linear_model,naive_bayes
from sklearn.cross_validation import train_test_split,cross_val_score,KFold,LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,cohen_kappa_score, make_scorer,r2_score
from sklearn.decomposition import FastICA,PCA,SparsePCA,NMF
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel

from unbalanced_dataset import SMOTE, SMOTETomek, NearMiss, ClusterCentroids, SMOTEENN, EasyEnsemble, BalanceCascade, OverSampler
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
#import spams
#import sys


#######################################################################################################################
def UnderSampling(method,verbose,X,y):
	if (method == 'cluster_cen'):
	# 'Clustering centroids'
		CC = ClusterCentroids(verbose=verbose)
		downX, downy = CC.fit_transform(X,y)
	if (method == 'NearMiss1'):	
	# 'NearMiss-1'
		NM1 = NearMiss(version=1, verbose=verbose)
		downX, downy = NM1.fit_transform(X,y)
	At_Risk_no_down = sum(downy)
	Typical_no_down = len(downy)-sum(downy)
	print "After Synthesis Data Undersampling. Train Typical number=%d, Train At Risk number=%d" %(Typical_no_down, At_Risk_no_down)	
	return downX,downy

def OverSampling(method,verbose,X,y,ratio):
	if (method == 'Random'):
		# 'Random over-sampling'
		OS = OverSampler(ratio=ratio, verbose=verbose)
		uppX, uppy = OS.fit_transform(X,y)

	if (method == 'SMOTE'):
		# 'SMOTE' DATA SYNTHESIS		
		smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
		uppX, uppy = smote.fit_transform(X,y)
		
	if (method == 'SVM_SMOTE'):
		# 'SMOTE' DATA SYNTHESIS
		svm_args={'class_weight' : 'auto'}
		svmsmote = SMOTE(ratio=ratio, verbose=verbose, kind='svm', **svm_args)
		uppX, uppy = svmsmote.fit_transform(X,y)		
	
	At_Risk_no_upp = sum(uppy)
	Typical_no_upp = len(uppy)-sum(uppy)
	print "After Synthesis Data Oversampling. Train Typical number=%d, Train At Risk number=%d" %(Typical_no_upp, At_Risk_no_upp)	
	return uppX,uppy


def OverUnder_Sampling(method,verbose,X,y,ratio):	
	if (method == 'SMOTE_TOMEK'):
		# 'SMOTE Tomek links'
		STK = SMOTETomek(ratio=ratio, verbose=verbose)
		UppDownX, UppDowny = STK.fit_transform(X,y)
	if (method == 'SMOTE_ENN'):
		# 'SMOTE ENN'
		SENN = SMOTEENN(ratio=ratio, verbose=verbose)
		UppDownX, UppDowny = SENN.fit_transform(X,y)

	At_Risk_no_UppDown = sum(UppDowny)
	Typical_no_UppDown = len(UppDowny)-sum(UppDowny)
	print "After Synthesis Data Oversampling. Train Typical number=%d, Train At Risk number=%d" %(Typical_no_UppDown, At_Risk_no_UppDown)	
	return UppDownX,UppDowny

def EnsembleSampling(method,verbose,X,y):
	if (method == 'EasyEnsemble'):
		EE = EasyEnsemble(verbose=verbose)
		ensembleX, ensembley = EE.fit_transform(X,y)		
	if (method == 'BalanceCascade'):
		BS = BalanceCascade(verbose=verbose)
		ensembleX, ensembley = BS.fit_transform(X,y)		
	print ensembley
	#At_Risk_no_ensemble = sum(ensembley)
	#Typical_no_ensemble = len(ensembley)-sum(ensembley)
	#print "After Synthesis Data Oversampling. Train Typical number=%d, Train At Risk number=%d" %(Typical_no_ensemble, At_Risk_no_ensemble)	
	return ensembleX,ensembley
def DimensionReduction(method,X,X_train,y_train,dimension,UnLabledData):
	Train_and_UnLabeled=np.vstack((X_train,UnLabledData))
	if (method == 'pca'):
		#Principle Component Analysis
		#The PCA(0 without indicating the number of components will chose: n_components=min(n_features,n_samples)		
		#pca = PCA()	
		#Xy=np.concatenate((X,y),axis=1)
		#pca.fit(X_train)
		#X_train = pca.transform(X_train)
		#To be able to chose number of componets automaticly Minca's method we need n_features<=n_samples
		#pca = PCA(n_components='mle')
		pca = PCA(n_components=dimension)
		pca.fit(Train_and_UnLabeled)
		X = pca.transform(X)
	if (method == 'lda'):
		#LDA dimention reduction number of diemnsion can be maximum (number of classes-1) 
		lda=LinearDiscriminantAnalysis(n_components=dimension)	
		lda.fit(X_train,y_train)
		X = lda.transform(X)
	if (method == 'pls'):
		pls=PLSRegression(n_components=dimension)	
		pls.fit(X_train,y_train)
		X = pls.transform(X)
	if (method == 'sparsepca'):
		#Sparse Principle Component Analysis
		spca = SparsePCA(n_components=dimension)
		spca.fit(Train_and_UnLabeled)
		X = spca.transform(X)

	if (method == 'ica'):
		#Indipendent Component Analysis			
		ica =FastICA(n_components=dimension)
		ica.fit(Train_and_UnLabeled)
		X = ica.transform(X)

	if (method == 'nmf'):

		#Non-Negative Matrix Factorization			
		####NMF using spams		
		#(U,V) = spams.nmf( np.transpose(X_train), return_lasso = True, K = dimension,iter=-5 )
		#mapped_X = X.dot(U)
		#X=mapped_X
		nmf=NMF(n_components = dimension)			
		nmf.fit(Train_and_UnLabeled)
		X = nmf.transform(X)

	if (method == 'ridge'):
	# Reconstruction with L2 (Ridge) penalization
		#alpha defines the amount of shrinkage
		alpha_range=np.logspace(-5,3,30)
		ridge = linear_model.RidgeCV(alphas=alpha_range)
		ridge.fit(X_train, y_train)
		print ridge.alpha_
		clf = linear_model.Ridge(alpha=ridge.alpha_)
		clf.fit(X_train, y_train)
		masked_coef = np.ma.masked_greater(clf.coef_,0).mask
		print clf.coef_
		#clf.coef_[clf.coef_>0] = 1
		#clf.coef_[clf.coef_<=0] = 0
		X=X[:,masked_coef]

	if (method == 'lasso'):
	# Reconstruction with L1 (Lasso) penalization
	# the best value of alpha was determined using cross validation with LassoCV
		alpha_range=np.logspace(-5,0,30)
		lasso = linear_model.LassoCV(alphas=alpha_range,positive=True)
		lasso.fit(X_train,y_train)
		print lasso.alpha_
		clf=linear_model.Lasso(alpha=lasso.alpha_,positive=True)
		clf.fit(X_train,y_train)
		masked_coef = np.ma.masked_greater(clf.coef_,0).mask
		print clf.coef_
		X=X[:,masked_coef]
	if (method == 'elasticNet'):		
	# ElasticNet
		alpha_range=np.logspace(-5,0,30)
		l1_ratio_range=[.1, .5, .7, .9, .95, .99, 1]
		elnet = linear_model.ElasticNetCV(alphas=alpha_range,l1_ratio=l1_ratio_range,positive=True)
		elnet.fit(X_train, y_train)
		print elnet.alpha_,elnet.l1_ratio_
		#clf = linear_model.ElasticNet(alpha=0.5,l1_ratio=1,positive=True)
		clf = linear_model.ElasticNet(alpha=elnet.alpha_,l1_ratio=elnet.l1_ratio_,positive=True)
		clf.fit(X_train, y_train)
		masked_coef = np.ma.masked_greater(clf.coef_,0).mask
		print clf.coef_
		X=X[:,masked_coef]
	if (method == 'featSel'):
		#'LINEAR SVM'
		linear_svm = svm.LinearSVC(penalty="l1", dual=False)
		C_range=np.logspace(-1, 1, 3)
		param_grid = {'C':C_range}
		grid = GridSearchCV(linear_svm, param_grid, verbose=1)
		grid.fit(X_train, y_train)
		clf=grid.best_estimator_

		"""#'ELASTIC NET'
		alpha_range=np.logspace(-5,0,30)
		l1_ratio_range=[.1, .5, .7, .9, .95, .99, 1]
		elnet = linear_model.ElasticNetCV(alphas=alpha_range,l1_ratio=l1_ratio_range,positive=True)
		elnet.fit(X_train, y_train)
		print elnet.alpha_,elnet.l1_ratio_
		clf = linear_model.ElasticNet(alpha=elnet.alpha_,l1_ratio=elnet.l1_ratio_,positive=True)
		clf.fit(X_train, y_train)
		print clf.coef_"""

		"""#'LASSO'
		alpha_range=np.logspace(-5,0,30)
		lasso = linear_model.LassoCV(alphas=alpha_range,positive=True)
		lasso.fit(X_train,y_train)
		print lasso.alpha_
		clf=linear_model.Lasso(alpha=lasso.alpha_,positive=True)
		clf.fit(X_train,y_train)
		print clf.coef_"""

		"""#'RIDGE'
		alphas=np.logspace(-1,1,30)
		ridge = linear_model.RidgeCV(alphas=alphas)
		ridge.fit(X_train, y_train)
		clf = linear_model.Ridge(alpha=ridge.alpha_)
		clf.fit(X_train, y_train)
		#print clf.coef_"""


		model = SelectFromModel(clf,prefit=True)
		X = model.transform(X)
		print model.threshold_
		
	if (method == 'nnsc'):
		#non-negative sparse coding	
		(U,V) = spams.nnsc( np.transpose(Train_and_UnLabeled), return_lasso = True, K = dimension)
		mapped_X = X.dot(U)
		X=mapped_X                                    
	return X

def Parameter_tunning(method,model,X_train,y_train,X_test,y_test):

	print "Perfoming hyper-parameters Tunning ..."
	kappa_scorer = make_scorer(cohen_kappa_score)
	#scoring=kappa_scorer
	scoring = 'precision_weighted'
	#scoring = 'accuracy'
	
	#C_range = np.logspace(-3,5, 9)
	C_range = np.logspace(-1, 1, 3)
	gamma_range = np.logspace(-5,3, 9)
	k_range = range(1, 31)
	#degree_range=range(1,6)
	degree_range=range(2,4)
	weight_options = ['uniform', 'distance']
	
	if (method=='knn'):
		param_grid = dict(n_neighbors=k_range, weights=weight_options)
	if (method=='linear_svm'):
		param_grid = {'C':C_range}
	if (method=='rbf_svm'):
		param_grid = {'C':C_range,'gamma':gamma_range}
	if (method=='poly_svm'):
		param_grid = {'C':C_range,'degree':degree_range}
	if (method=='lr'):
		param_grid = {'C':C_range}
	if (method=='dt'):
		param_grid = {#"criterion": ["gini", "entropy"], 
			      #"min_samples_split": [2, 10, 20], 
			      #"max_depth": [None, 2, 5, 10], 
			      #"min_samples_leaf": [1, 5, 10], 
			      "max_leaf_nodes": [None, 5, 10, 20]
			     }
	if (method=='rfc'):
		"""param_grid = {#"n_estimators": [200,700],
              		      #"max_features": ['auto', 'sqrt', 'log2'],
              		      "max_features": [1, 3, 10]	              
		              #"bootstrap": [True, False],              		      
		              }"""

		param_grid = {#"criterion": ["gini", "entropy"], 
			      #"min_samples_split": [2, 10, 20], 
			      #"max_depth": [None, 2, 5, 10], 
			      #"min_samples_leaf": [1, 5, 10], 
			      "max_leaf_nodes": [None, 5, 10, 20]			    
			     }
	#verbose : integer, Controls the verbosity: the higher, the more messages, default is 0.
	grid = GridSearchCV(model, param_grid, verbose=1,scoring=scoring)
	grid.fit(X_train, y_train)
	print "Result of tunning on Training Data is:"
	print "Best %s = %f, best parametrs are:" %(scoring , grid.best_score_), grid.best_params_
	
	return grid
	
def print_result(expected,predicted):
	cm=confusion_matrix(expected, predicted)
	#panda_cm=pd.crosstab(expected, predicted)
	TN=cm[0][0]
	FP=cm[0][1]
	FN=cm[1][0]	
	TP=cm[1][1]

	accuracy = float(TP+TN)/np.sum(cm)
	sensivity = float(TP)/(TP+FN)
	specificity = float(cm[0][0])/np.sum(cm[0])

	ppv = (float(TP)/(TP+FP)) if TP!=0 else 0
	npv = float(TN)/(FN+TN)	if TN!=0 else 0	
	
	return accuracy,sensivity,specificity,ppv,npv,cm
	"""print(classification_report(expected, predicted))
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]	
	print "kappa score=%f" %(cohen_kappa_score(expected, predicted))
	print "Accuracy =%f and Sensitivity=%f and Specificity=%f and PPV=%f and NPV=%f" %(accuracy,sensivity,specificity,ppv,npv)
	print 'Confusion Matrix: '
	print np.matrix(cm)
	print 'Normalized Confusion Matrix: '
	print np.matrix(cm_normalized)"""

def print_kfold(method,accuracy,sensitivity,specificity,ppv,npv,cm,no_folds):
	print '--------------------------------------------------------'
	print method," Accuracy: ",accuracy/float(no_folds)
	print method," Sensitivity: ",sensitivity/float(no_folds)
	print method," Specificity: ",specificity/float(no_folds)
	print method," Positive Predictive Value: ",ppv/float(no_folds)
	print method," Negative Predictive Value: ",npv/float(no_folds)
	print method," Confusion Matrix: "
	print cm

def statistical_analysis_DataFrame(df):
	groupby_label=df.groupby('Label')
	print groupby_label.mean()
	caudate_cols = [col for col in df.columns if 'caudate' in col.strip().lower()]
	cingulum_cols = [col for col in df.columns if 'cingul' in col.strip().lower()]
	frontal_cols = [col for col in df.columns if '_frontal' in col.strip().lower()]

	groupby_label.boxplot(column=caudate_cols)
	groupby_label.boxplot(column=cingulum_cols)
	groupby_label.boxplot(column=frontal_cols)

	#groupby_label.boxplot(column=df.columns)
	plt.show()
	

	colStr='Label~Caudate_Left'
	for col in df.columns:
		if str(col.strip()) not in ['Caudate_Left','Label']: 
			colStr+="+"
			colStr+=str(col.strip())
	#print colStr
	lm=smf.ols(formula=colStr,data=df).fit()
	print lm.summary()
	
def statistical_analysis(X,y):	
	#X = df.ix[:,df.columns!='Label']
	#y = df['Label']

	## fit a OLS model with intercept on all columns
	X = sm.add_constant(X)
	est = sm.OLS(y, X).fit()

	print est.summary()
#######################################################################################################################
#INITIALIZING
knn_accuracy=knn_sensivity=knn_specificity=knn_ppv=knn_npv=knn_cm=0
lr_accuracy=lr_sensivity=lr_specificity=lr_ppv=lr_npv=lr_cm=0
nb_accuracy=nb_sensivity=nb_specificity=nb_ppv=nb_npv=nb_cm=0
dt_accuracy=dt_sensivity=dt_specificity=dt_ppv=dt_npv=dt_cm=0
rfc_accuracy=rfc_sensivity=rfc_specificity=rfc_ppv=rfc_npv=rfc_cm=0
rbf_accuracy=rbf_sensivity=rbf_specificity=rbf_ppv=rbf_npv=rbf_cm=0
linear_accuracy=linear_sensivity=linear_specificity=linear_ppv=linear_npv=linear_cm=0
poly_accuracy=poly_sensivity=poly_specificity=poly_ppv=poly_npv=poly_cm=0

no_folds=10
UnLabledData=pd.read_csv('./neo_files/neo_BrainData.csv')

DataF = pd.read_csv('./neo_files/neo_Brain_cog.csv')
no_dims =60

#DataF = pd.read_csv('./neo_files/neo_BrainSubsetFeats.csv')
#no_dims =20

#dimred_method='pls'
#dimred_method='nmf'
#dimred_method='pca'
#dimred_method='lasso'
#dimred_method='ridge'
#dimred_method='elasticNet'
dimred_method='featSel'
#dimred_method='nnsc'
#dimred_method='lda'
#dimred_method='sparsepca'
#dimred_method='ica'

#######################################################################################################################


UnLabledData=UnLabledData.loc[UnLabledData['subjectID'].isin(DataF['subjectID'])==False]


DataF = DataF.ix[:,DataF.columns!='subjectID']
column_header = 'Label'

statistical_analysis_DataFrame(DataF)


Data = DataF.ix[:,DataF.columns!=column_header]
labels = DataF[column_header]

At_Risk_no = sum(labels)
Typical_no = len(labels)-sum(labels)
print "Typical number=%d, At Risk number=%d" %(Typical_no, At_Risk_no)
label_names= ['Typical','At_Risk']

UnLabledData=UnLabledData.ix[:,UnLabledData.columns!='subjectID']
UnLabledData=np.array(UnLabledData)
#X,y=(Data.values).tolist(),labels.tolist()
X,y=np.array((Data.values)),np.array(labels)
#y = y[:,np.newaxis]

print "X=",X.shape,'Unlabled=',UnLabledData.shape

#Normalization rescale the data between 0 and 1
#norm can be 'l1','l2','max'
#X=preprocessing.normalize(X,norm='l2')
#UnLabledData=preprocessing.normalize(UnLabledData,norm='l2')

#Standardization 
#Shift the distribution to have mean of zero and standard devisation of 1 (unit Variance)
X=preprocessing.scale(X)
UnLabledData=preprocessing.scale(UnLabledData)
#Splitting
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_validation.cross_val_score(
#clf, iris.data, iris.target, cv=5)



kf = KFold(len(y), n_folds=no_folds)
#loo = LeaveOneOut(len(y))
print kf
#print loo
for k,(train_index, test_index) in enumerate(kf):
#for k,(train_index, test_index) in enumerate(loo):
	print '\n\n...........................................performing Fold #%d...............................................' %(k+1)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	X_new = DimensionReduction(dimred_method, X, X_train, y_train, no_dims,UnLabledData)
	print "Dimensions after ",dimred_method," = ", X_new.shape
	X_train, X_test = X_new[train_index], X_new[test_index]	

	statistical_analysis(X_train,y_train)
	
	At_Risk_no_train = sum(y_train)
	Typical_no_train = len(y_train)-sum(y_train)
	print "Typical number Train=%d, At Risk number Train =%d" %(Typical_no_train, At_Risk_no_train)

	At_Risk_no_test = sum(y_test)
	Typical_no_test = len(y_test)-sum(y_test)
	print "Typical number Test=%d, At Risk number Test =%d" %(Typical_no_test, At_Risk_no_test)

	ratio = Typical_no_train/At_Risk_no_train-1
	#ratio = float(At_Risk_no_train)/float(Typical_no_train)
	#ratio = 5.0
	X_train,y_train = OverSampling('SMOTE',False,X_train,y_train,ratio)
	#X_train,y_train = UnderSampling('cluster_cen',False,X_train,y_train)
	#X_train,y_train = OverUnder_Sampling('SMOTE_TOMEK',False,X_train,y_train,ratio)
	#X_train,y_train = EnsembleSampling('EasyEnsemble',False,X_train,y_train)
	print 
	print "\n--- Classification Results ---"
	print "K-NEAREST NEIGHBOR"
	
	print X_train.shape	

	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	expected=y_test
	predicted =knn.predict(X_test)
	
	[knn_accuracy_,knn_sensivity_,knn_specificity_,knn_ppv_,knn_npv_,knn_cm_] = print_result(expected,predicted)

	grid=Parameter_tunning('knn',knn,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)

	[knn_accuracy_,knn_sensivity_,knn_specificity_,knn_ppv_,knn_npv_,knn_cm_] = print_result(expected,predicted)
	knn_accuracy+=knn_accuracy_
	knn_sensivity+=knn_sensivity_
	knn_specificity+=knn_specificity_
	knn_ppv+=knn_ppv_
	knn_npv+=knn_npv_
	knn_cm+=knn_cm_

	print "***"
	print "LINEAR REGRESSION"
	lr = linear_model.LogisticRegression()
	lr.fit(X_train, y_train)
	expected=y_test
	predicted =lr.predict(X_test)

	[lr_accuracy_,lr_sensivity_,lr_specificity_,lr_ppv_,lr_npv_,lr_cm_] = print_result(expected,predicted)

	grid=Parameter_tunning('lr',lr,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)

	[lr_accuracy_,lr_sensivity_,lr_specificity_,lr_ppv_,lr_npv_,lr_cm_] = print_result(expected,predicted)
	lr_accuracy+=lr_accuracy_
	lr_sensivity+=lr_sensivity_
	lr_specificity+=lr_specificity_
	lr_ppv+=lr_ppv_
	lr_npv+=lr_npv_
	lr_cm+=lr_cm_

	print "***"
	print "NAIVE BAYES"
	nb = naive_bayes.GaussianNB()
	nb.fit(X_train, y_train)
	expected=y_test
	predicted =nb.predict(X_test)
	
	[nb_accuracy_,nb_sensivity_,nb_specificity_,nb_ppv_,nb_npv_,nb_cm_] = print_result(expected,predicted)
	nb_accuracy+=nb_accuracy_
	nb_sensivity+=nb_sensivity_
	nb_specificity+=nb_specificity_
	nb_ppv+=nb_ppv_
	nb_npv+=nb_npv_
	nb_cm+=nb_cm_

	print "***"
	print "DECISION TREE"
	dt = DecisionTreeClassifier()
	dt.fit(X_train, y_train)
	expected=y_test
	predicted =dt.predict(X_test)

	[dt_accuracy_,dt_sensivity_,dt_specificity_,dt_ppv_,dt_npv_,dt_cm_] = print_result(expected,predicted)
	
	grid=Parameter_tunning('dt',dt,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)
	[dt_accuracy_,dt_sensivity_,dt_specificity_,dt_ppv_,dt_npv_,dt_cm_] = print_result(expected,predicted)

	dt_accuracy+=dt_accuracy_
	dt_sensivity+=dt_sensivity_
	dt_specificity+=dt_specificity_
	dt_ppv+=dt_ppv_
	dt_npv+=dt_npv_
	dt_cm+=dt_cm_

	print "***"
	print "RANDOM FOREST"
	rfc = RandomForestClassifier() 
	rfc.fit(X_train, y_train)
	expected=y_test
	predicted =rfc.predict(X_test)
	
	[rfc_accuracy_,rfc_sensivity_,rfc_specificity_,rfc_ppv_,rfc_npv_,rfc_cm_] = print_result(expected,predicted)
	
	grid=Parameter_tunning('rfc',rfc,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)
	[rfc_accuracy_,rfc_sensivity_,rfc_specificity_,rfc_ppv_,rfc_npv_,rfc_cm_] = print_result(expected,predicted)

	rfc_accuracy+=rfc_accuracy_
	rfc_sensivity+=rfc_sensivity_
	rfc_specificity+=rfc_specificity_
	rfc_ppv+=rfc_ppv_
	rfc_npv+=rfc_npv_
	rfc_cm+=rfc_cm_
	
	print "***"
	print "SVM WITH RBF KERNEL"
	rbf_svm = svm.SVC(kernel='rbf')
	rbf_svm.fit(X_train, y_train)
	expected=y_test
	predicted =rbf_svm.predict(X_test)
	
	[rbf_accuracy_,rbf_sensivity_,rbf_specificity_,rbf_ppv_,rbf_npv_,rbf_cm_] = print_result(expected,predicted)
	
	grid=Parameter_tunning('rbf_svm',rbf_svm,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)
	[rbf_accuracy_,rbf_sensivity_,rbf_specificity_,rbf_ppv_,rbf_npv_,rbf_cm_] = print_result(expected,predicted)

	rbf_accuracy+=rbf_accuracy_
	rbf_sensivity+=rbf_sensivity_
	rbf_specificity+=rbf_specificity_
	rbf_ppv+=rbf_ppv_
	rbf_npv+=rbf_npv_
	rbf_cm+=rbf_cm_
	
	print "***"
	print "SVM WITH LINEAR KERNEL"
	linear_svm = svm.LinearSVC()
	linear_svm.fit(X_train, y_train)
	expected=y_test
	predicted =linear_svm.predict(X_test)
	
	[linear_accuracy_,linear_sensivity_,linear_specificity_,linear_ppv_,linear_npv_,linear_cm_] = print_result(expected,predicted)
	
	grid=Parameter_tunning('linear_svm',linear_svm,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)
	[linear_accuracy_,linear_sensivity_,linear_specificity_,linear_ppv_,linear_npv_,linear_cm_] = print_result(expected,predicted)

	linear_accuracy+=linear_accuracy_
	linear_sensivity+=linear_sensivity_
	linear_specificity+=linear_specificity_
	linear_ppv+=linear_ppv_
	linear_npv+=linear_npv_
	linear_cm+=linear_cm_

	print "***"
	print "SVM WITH POLYNOMIAL KERNEL"
	#poly_svm = svm.SVC(kernel='poly',verbose=True)
	#tol=1e-3 is the defult value and shows the tolerance
	poly_svm = svm.SVC(kernel='poly',max_iter=-1,tol=1e-3)
	poly_svm.fit(X_train, y_train)
	expected=y_test
	predicted =poly_svm.predict(X_test)
	
	[poly_accuracy_,poly_sensivity_,poly_specificity_,poly_ppv_,poly_npv_,poly_cm_] = print_result(expected,predicted)

	grid=Parameter_tunning('poly_svm',poly_svm,X_train,y_train,X_test,y_test)
	#print "Results after parameter Tunning:"
	predicted =grid.predict(X_test)
	[poly_accuracy_,poly_sensivity_,poly_specificity_,poly_ppv_,poly_npv_,poly_cm_] = print_result(expected,predicted)
	
	poly_accuracy+=poly_accuracy_
	poly_sensivity+=poly_sensivity_
	poly_specificity+=poly_specificity_
	poly_ppv+=poly_ppv_
	poly_npv+=poly_npv_
	poly_cm+=poly_cm_
	

print_kfold('K-Nearest Neighbor',knn_accuracy,knn_sensivity,knn_specificity,knn_ppv,knn_npv,knn_cm,no_folds)
print_kfold('Linear Regression',lr_accuracy,lr_sensivity,lr_specificity,lr_ppv,lr_npv,lr_cm,no_folds)
print_kfold('Naive Bays',nb_accuracy,nb_sensivity,nb_specificity,nb_ppv,nb_npv,nb_cm,no_folds)
print_kfold('Decision Tree',dt_accuracy,dt_sensivity,dt_specificity,dt_ppv,dt_npv,dt_cm,no_folds)
print_kfold('Random Forest',rfc_accuracy,rfc_sensivity,rfc_specificity,rfc_ppv,rfc_npv,rfc_cm,no_folds)
print_kfold('RBF Kernel SVM',rbf_accuracy,rbf_sensivity,rbf_specificity,rbf_ppv,rbf_npv,rbf_cm,no_folds)
print_kfold('Linear Kernel SVM',linear_accuracy,linear_sensivity,linear_specificity,linear_ppv,linear_npv,linear_cm,no_folds)
print_kfold('Polynomial Kernel SVM',poly_accuracy,poly_sensivity,poly_specificity,poly_ppv,poly_npv,poly_cm,no_folds)
