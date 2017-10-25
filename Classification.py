import itertools
import pylab as pl
from os import system
from sklearn import preprocessing, svm, linear_model, naive_bayes
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA, PCA, SparsePCA, NMF
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from frameworks.CPLELearning import CPLELearningModel
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

from unbalanced_dataset import SMOTE, SMOTETomek, NearMiss, ClusterCentroids, SMOTEENN, EasyEnsemble, BalanceCascade, OverSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm


#######################################################################################################################
def make_binary(D, L, S):
    hr_lbs = np.where(L == 1)
    lr_lbs = np.where(L == 0)

    B = [];
    S = [];

    m = np.zeros(D.shape[1])
    for i in range(D.shape[1]):
        d = D[:, i]
        if not S:
            m_hr = np.median(d[hr_lbs])
            m_lr = np.median(d[lr_lbs])

            m[i] = (m_hr + m_lr) / 2

            idxg = np.where(d >= m[i])
            idxl = np.where(d < m[i])
        else:
            idxg = np.where(d >= S[i])
            idxl = np.where(d < S[i])
        d[idxg] = 1
        d[idxl] = 0
        if len(B) == 0:
            B = d
        else:
            B = np.c_[B, d]

    if not S:
        S = m
    return B, S


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_confusion_matrix(C, class_labels=['0', '1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """

    assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0];
    fp = C[0, 1];
    fn = C[1, 0];
    tp = C[1, 1];

    NP = fn + tp  # Num positive examples
    NN = tn + fp  # Num negative examples
    N = NP + NN

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Negative: %d\n(Num Neg: %d)' % (tn, NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Negative: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Positive: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Positive: %d\n(Num Pos: %d)' % (tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'Specificity: %.2f' % (tn / (fp + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'Sensitivity: %.2f' % (tp / (tp + fn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / N),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 2,
            'NPV: %.2f' % (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'PPV: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig('results/Diagrams/confusion_matrix.eps', format='eps', dpi=1000)
    plt.show()


def show_bar_diagram(n_groups, Acc_StructMRI, Acc_DTI,Acc_FF, Acc_FF_G):
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 1

    rects1 = plt.bar(index, 100 * Acc_StructMRI, bar_width,
                     alpha=opacity,
                     color='b',
                     label='StructralMRI',
                     align='center')

    rects2 = plt.bar(index + bar_width, 100 * Acc_DTI, bar_width,
                     alpha=opacity,
                     color='g',
                     label='DTI',
                     align='center')
    rects3 = plt.bar(index + 2 * bar_width, 100 * Acc_FF, bar_width,
                     alpha=opacity,
                     color='y',
                     label='Structral + DTI',
                     align='center')

    rects4 = plt.bar(index + 3 * bar_width, 100 * Acc_FF_G, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Structral + DTI + Gestational Age',
                     align='center')

    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.xticks(index + bar_width, ('Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV'))
    plt.yticks()
    plt.legend(loc=4, prop={'size': 12})
    ax.set_ylabel('Percent Recognition (%)')
   # ax.set_xlabel('Measures')
    plt.tight_layout()
    plt.savefig('results/Diagrams/comparison.eps', format='eps', dpi=1000)
    plt.show()

def select_classifier(binary, cls_method, X_train, y_train, X_test, y_test, subjects_test, ver, UnlabeledData):
    if binary:
        [X_train, S] = make_binary(X_train, y_train, [])
        [X_test, S] = make_binary(X_test, y_test, S)
    expected = y_test
    aligned_subs = subjects_test
    confidence_scores = []

    if (cls_method == 'knn'):
        if print_folds:
            print "\n--- Classification Results ---"
            print "K-NEAREST NEIGHBOR"

        clsf = KNeighborsClassifier()
        confidence_scores = [1.0] * len(X_test)

    if (cls_method == 'lr'):
        method_name = 'Logistic Regression'
        if print_folds:
            print "***"
            print "LINEAR REGRESSION"

        clsf = linear_model.LogisticRegression(random_state=2)
        conf_score = True

    # confidence_scores = cls.predict_proba(X_test)

    if (cls_method == 'mlp'):
        method_name = 'deep neural net'
        if print_folds:
            print "***"
            print "MultiLayer Perception(NN)"

        clsf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1, alpha=1)
        conf_score = False
    # solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1

    if (cls_method == 'nb'):
        method_name = 'Naive Bays'
        if print_folds:
            print "***"
            print "NAIVE BAYES"

        clsf = naive_bayes.GaussianNB()
        conf_score = False
    # print map(lambda element:element[-1],confidence_scores)
    # print zip(*confidence_scores)

    if (cls_method == 'dt'):
        method_name = 'Decision Tree'
        if print_folds:
            print "***"
            print "DECISION TREE"

        clsf = DecisionTreeClassifier(max_depth=5)
        conf_score = False

    if (cls_method == 'rfc'):
        method_name = 'Random Forest'
        if print_folds:
            print "***"
            print "RANDOM FOREST"

        clsf = RandomForestClassifier(n_estimators=100)
        conf_score = False

    if (cls_method == 'rbf'):
        method_name = 'SVM(RBF)'
        if print_folds:
            print "***"
            print "SVM WITH RBF KERNEL"

        clsf = svm.SVC(kernel='rbf')
        conf_score = True

    if (cls_method == 'linear_svm'):
        method_name = 'SVM(Linear)'
        if print_folds:
            print "***"
            print "SVM WITH LINEAR KERNEL"

        clsf = svm.LinearSVC()
        conf_score = True

    if (cls_method == 'poly_svm'):
        method_name = 'SVM(poly)'
        if print_folds:
            print "***"
            print "SVM WITH POLYNOMIAL KERNEL"

        # poly_svm = svm.SVC(kernel='poly',verbose=True)
        # tol=1e-3 is the defult value and shows the tolerance
        clsf = svm.SVC(kernel='poly', max_iter=-1, tol=1e-3)
        conf_score = True
    if (cls_method == 'semi'):
        method_name = 'semi-supervised'
        if print_folds:
            print "***"
            print "Semi-supervised"
        [clsf, predicted] = semi_supervised(X_train, y_train, X_test, y_test, UnlabeledData)
        conf_score = False

    if cls_method != 'semi':
        clsf.fit(X_train, y_train)
        predicted = clsf.predict(X_test)
    [cls_accuracy_, cls_sensivity_, cls_specificity_, cls_ppv_, cls_npv_, cls_cm_] = print_result(expected, predicted)

    if cls_method != 'semi':
        grid = Parameter_tunning(cls_method, clsf, X_train, y_train, X_test, y_test, ver, print_folds, scoring)
        # print "Results after parameter Tunning:"
        predicted = grid.predict(X_test)
    if not confidence_scores:
        if conf_score:
            confidence_scores = FindConfScore_DecFunc(X_test, expected, predicted, clsf, method_name, print_confidence)
        else:
            confidence_scores = np.zeros(len(predicted))
            cs_probList = clsf.predict_proba(X_test)
            for inx in range(len(predicted)):

                if predicted[inx] == 0:
                    confidence_scores[inx] = cs_probList[inx, 0]
                else:
                    confidence_scores[inx] = cs_probList[inx, 1]

    # print_confidenceScore(method_name,expected,predicted,confidence_scores)


    [cls_accuracy_, cls_sensivity_, cls_specificity_, cls_ppv_, cls_npv_, cls_cm_] = print_result(expected, predicted)

    return method_name, cls_accuracy_, cls_sensivity_, cls_specificity_, cls_ppv_, cls_npv_, cls_cm_, predicted, expected, aligned_subs, confidence_scores


def semi_supervised(X_train, y_train, X_test, y_test, unlabeled):
    # y_train = [element if element else 0 for element in y_train]
    y_train[y_train == -1] = 0
    # instead of just using ys= y_train we use the built in fuction list so the changes of ys doesn't cahnge y_train
    ys = list(y_train)
    ys = ys + list(np.array([-1] * len(unlabeled)))

    # for inx in range(len(unlabeled)):
    #	ys += [-1]
    print 'x_train', X_train.shape
    print 'unlabled', unlabeled.shape
    X = np.vstack((X_train, unlabeled))

    # supervised score
    # basemodel = WQDA() # weighted Quadratic Discriminant Analysis

    basemodel = SGDClassifier(loss='log', penalty='l1')  # scikit logistic regression
    basemodel.fit(X_train, y_train)

    # print "supervised log.reg. score", basemodel.score(X_test, y_test)

    X = np.array(X)
    ys = np.array(ys)

    # fast (but naive, unsafe) self learning framework
    # Semi-supervised learning using gaussian fields and harmonic functions
    ##### ICML 2003 #####
    """ssmodel = SelfLearningModel(basemodel)

    ssmodel.fit(X, ys)
    print "self-learning log.reg. score", ssmodel.score(X_test, y_test)"""

    # semi-supervised score (base model has to be able to take weighted samples)
    # Contrastive Pessimistic Likelihood Estimation for Semi-Supervised Classification
    # ssmodel = scikitTSVM.SKTSVM(kernel='linear')
    # ssmodel.fit(X, ys.astype(int))

    #### SKlearn label propagation
    # ssmodel = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)

    ##### TPAMI 2015#####
    ssmodel = CPLELearningModel(basemodel, predict_from_probabilities=True)
    ssmodel.fit(X, ys)

    predicted = ssmodel.predict(X_test)
    # To change boolean(True, False) to scalar (0 , 1)
    predicted = 1 * predicted
    # confidence_scores = ssmodel.predict_proba(X_test)
    # print "CPLE semi-supervised log.reg. score", ssmodel.score(X_test, y_test)

    # semi-supervised score, RBF SVM model
    """ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True) # RBF SVM
    ssmodel.fit(X, ys)
    print "CPLE semi-supervised RBF SVM score", ssmodel.score(X_test, y_test)"""
    return ssmodel, predicted


def print_result(expected, predicted):
    if np.array_equal(expected, predicted):
        cm = np.zeros((2, 2), dtype=int)
        if sum(expected) == 0:
            cm[0][0] = 3
            cm[0][1] = 0
            cm[1][0] = 0
            cm[1][1] = 0
        else:
            cm[0][0] = 0
            cm[0][1] = 0
            cm[1][0] = 0
            cm[1][1] = 3
    else:
        cm = confusion_matrix(expected, predicted)

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    accuracy = float(TP + TN) / np.sum(cm)
    sensivity = float(TP) / (TP + FN) if (TP + FN) != 0 else 0
    specificity = float(cm[0][0]) / np.sum(cm[0]) if np.sum(cm[0]) != 0 else 0

    ppv = (float(TP) / (TP + FP)) if TP != 0 else 0
    npv = float(TN) / (FN + TN) if TN != 0 else 0

    return accuracy, sensivity, specificity, ppv, npv, cm


def show_Results(method, accuracy, sensitivity, specificity, ppv, npv, cm, no_folds):
    print '--------------------------------------------------------'
    print method, " Accuracy: ", accuracy / float(no_folds)
    print method, " Sensitivity: ", sensitivity / float(no_folds)
    print method, " Specificity: ", specificity / float(no_folds)
    print method, " Positive Predictive Value: ", ppv / float(no_folds)
    print method, " Negative Predictive Value: ", npv / float(no_folds)
    print method, " Confusion Matrix: "
    print cm


def print_kfold(method, confidence_score, accuracy, sensitivity, specificity, ppv, npv, cm, predicted, expected,
                aligned_subs, no_folds):
    show_Results(method, accuracy, sensitivity, specificity, ppv, npv, cm, no_folds)
    ####################################
    # build a dataframe for aligned subs and expected and predicted and return it
    # save for each classification method into a folder named results
    print "length of Subject ID=%d, Real_Label=%d , Predicted_label=%d, confidence score=%d" % (
    len(aligned_subs), len(expected), len(predicted), len(confidence_score))
    df = pd.DataFrame(columns=['subjectID', 'Real_Label', 'Predicted_Label'])
    df['subjectID'] = aligned_subs
    df['Real_Label'] = expected
    df['Predicted_Label'] = predicted
    df['confidence_score'] = confidence_score

    return df


def statistical_analysis_DataFrame(df):
    print df.columns
    groupby_label = df.groupby('Label')
    print groupby_label.mean()

    caudate_cols = [col for col in df.columns if 'caud' in col.strip().lower()]

    cingulum_cols = [col for col in df.columns if 'cingul' in col.strip().lower()]

    frontal_cols1 = [col for col in df.columns if ' frontal' in col.strip().lower()]
    frontal_cols2 = [col for col in df.columns if '_frontal' in col.strip().lower()]
    frontal_cols = frontal_cols1 + frontal_cols2

    groupby_label.boxplot(column=caudate_cols)
    groupby_label.boxplot(column=cingulum_cols)
    groupby_label.boxplot(column=frontal_cols)

    # groupby_label.boxplot(column=df.columns)
    plt.show()

    colStr = 'Label~ICV'
    for col in df.columns:
        if str(col.strip()) not in ['ICV', 'Label']:
            colStr += "+"
            colStr += str(col.strip())

    lm = smf.ols(formula=colStr, data=df).fit()
    print lm.summary()


def statistical_analysis(X, y, roi_names):
    R2 = []
    pval = []

    print('*************Multi-variable univariate analysis**************')
    # np.save('X_train_gfr',X)
    X = sm.add_constant(X)
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())
    print est2.pvalues[est2.pvalues < 0.01]
    print est2.rsquared
    print est2.params

    roi_n = len(roi_names)
    print "roi length", roi_n
    weights = est2.params[1:]

    weights_abs = np.absolute(weights)

    weights_sorted = np.sort(weights_abs)
    indices = np.argsort(weights_abs)

    pvals = est2.pvalues[1:]
    print pvals.shape
    pv = pvals[indices[::-1]]
    colors = []

    for i in range(roi_n):
        if pv[i] < 0.001:
            colors.append('lawngreen')
        else:
            colors.append('deepskyblue')

    hatching = []
    w = weights[indices[::-1]]
    print 'length w', len(w)
    for i in range(roi_n):
        if w[i] >= 0:
            hatching.append(' ')
        else:
            hatching.append('///')

    fig1, ax = plt.subplots()
    # fig1.tight_layout()
    ax.bar(range(weights_sorted.shape[0]), weights_abs[indices[::-1]], color=colors)
    # Loop over the bars
    for i, thisbar in enumerate(ax.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatching[i])
    ax.set_xticks(np.arange(0, (roi_n - 1), 1))
    roi_names = np.array(roi_names)
    print 'indices:', indices[3], type(indices), type(indices[3]), type(roi_names), type(weights)
    ax.set_xticklabels(roi_names[indices[::-1]], rotation='vertical')
    # ax.set_xlabel('AAL ROI')
    ax.set_ylabel('absolute weight')

    # ax2=ax.twinx()
    # ax2.plot(range(weights_sorted.shape[0]), pv, color='red')
    # ax2.set_ylabel('p-value', color='red')

    plt.tight_layout()
    plt.savefig("results/Diagrams/R2_cros.eps", format='eps', dpi=1000)
    plt.show()

    ## fit a OLS model with intercept on all columns
    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()

    print est.summary()


def visualize_corelation(DataF):
    names = DataF.columns
    correlations = DataF.corr()
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


def find_corrupted_digits(Data):
    print 'Start Fix Corrupted Data ...'
    for col in Data.columns:
        Data[col] = Data[col].astype(str)
    count_dot = Data[Data.columns].applymap(lambda x: str.count(x, '.'))
    for col in Data.columns:
        for inx in range(len(count_dot[col])):
            if (count_dot.iloc[inx][col] > 1):
                print "Fixed Corrupted Data"
                cell_value = Data.iloc[inx][col]
                cell_value = cell_value[0:cell_value.find('.', cell_value.find('.') + 1)]
                Data.iloc[inx][col] = cell_value
    for col in Data.columns:
        if col != 'subjectID':
            Data[col] = Data[col].astype(float)
    return Data


def change_col_name(df, prefix):
    cols = df.columns
    cols = cols.map(lambda x: prefix + x if (x != 'subjectID') else x)
    df.columns = cols
    return df


def UnderSampling(method, verbose, X, y):
    if (method == 'cluster_cen'):
        # 'Clustering centroids'
        CC = ClusterCentroids(verbose=verbose)
        downX, downy = CC.fit_transform(X, y)
    if (method == 'NearMiss1'):
        # 'NearMiss-1'
        NM1 = NearMiss(version=1, verbose=verbose)
        downX, downy = NM1.fit_transform(X, y)
    At_Risk_no_down = sum(downy)
    Typical_no_down = len(downy) - sum(downy)
    print "After Synthesis Data Undersampling: Train Typical number=%d, Train At Risk number=%d" % (
    Typical_no_down, At_Risk_no_down)
    return downX, downy


def OverSampling(method, verbose, X, y, ratio):
    if (method == 'Random'):
        # 'Random over-sampling'
        OS = OverSampler(ratio=ratio, verbose=verbose)
        uppX, uppy = OS.fit_transform(X, y)

    if (method == 'SMOTE'):
        # 'SMOTE' DATA SYNTHESIS
        smote = SMOTE(ratio=ratio, random_state=0,verbose=verbose, kind='regular')
        uppX, uppy = smote.fit_transform(X, y)

    if (method == 'SVM_SMOTE'):
        # 'SMOTE' DATA SYNTHESIS
        svm_args = {'class_weight': 'auto'}
        svmsmote = SMOTE(ratio=ratio, verbose=verbose, kind='svm', **svm_args)
        uppX, uppy = svmsmote.fit_transform(X, y)

    At_Risk_no_upp = sum(uppy)
    Typical_no_upp = len(uppy) - sum(uppy)
    print "After Synthesis Data Oversampling: Train Typical number=%d, Train At Risk number=%d" % (
    Typical_no_upp, At_Risk_no_upp)
    return uppX, uppy


def OverUnder_Sampling(method, verbose, X, y, ratio):
    if (method == 'SMOTE_TOMEK'):
        # 'SMOTE Tomek links'
        STK = SMOTETomek(ratio=ratio, verbose=verbose)
        UppDownX, UppDowny = STK.fit_transform(X, y)
    if (method == 'SMOTE_ENN'):
        # 'SMOTE ENN'
        SENN = SMOTEENN(ratio=ratio, verbose=verbose)
        UppDownX, UppDowny = SENN.fit_transform(X, y)

    At_Risk_no_UppDown = sum(UppDowny)
    Typical_no_UppDown = len(UppDowny) - sum(UppDowny)
    print "After Synthesis Data Oversampling: Train Typical number=%d, Train At Risk number=%d" % (
    Typical_no_UppDown, At_Risk_no_UppDown)
    return UppDownX, UppDowny


def EnsembleSampling(method, verbose, X, y):
    if (method == 'EasyEnsemble'):
        EE = EasyEnsemble(verbose=verbose)
        ensembleX, ensembley = EE.fit_transform(X, y)
    if (method == 'BalanceCascade'):
        BS = BalanceCascade(verbose=verbose)
        ensembleX, ensembley = BS.fit_transform(X, y)
    print ensembley
    # At_Risk_no_ensemble = sum(ensembley)
    # Typical_no_ensemble = len(ensembley)-sum(ensembley)
    # print "After Synthesis Data Oversampling. Train Typical number=%d, Train At Risk number=%d" %(Typical_no_ensemble, At_Risk_no_ensemble)
    return ensembleX, ensembley


def ShowSelectedFeat(sf, COEForSCORE, names, threshold, showFeat):
    # This line shortens all the function but print all of the scores not just non-zeros
    # print sorted(zip(map(lambda x: round(x, 4), COEForSCORE), names), reverse=True)
    if showFeat:
        featureCOEForSCOREs = map(lambda x: round(x, 4), COEForSCORE)
        featureCOEForSCOREList = zip(featureCOEForSCOREs, names)
    # To not change the order this restiction command should apply after the zip
    return featureCOEForSCOREList


def ReturnFeatsScores(featureCOEForSCOREList, threshold, showFeat, sf):
    # threshold = 0

    featureCOEForSCOREList = [f for f in featureCOEForSCOREList if f[0] > 0]
    featureCOEForSCOREList = map(lambda f: (f[1], f[0]), featureCOEForSCOREList)

    featureCOEForSCOREListDic = dict(featureCOEForSCOREList)
    featList = []
    featureCOEForSCOREList = list(featureCOEForSCOREListDic.values())
    for val in featureCOEForSCOREList:
        # return the key for value

        featKey = featureCOEForSCOREListDic.keys()[featureCOEForSCOREListDic.values().index(val)]
        featureCOEForSCOREListDic.pop(featKey)
        if 'long' in featKey:
            featKey = featKey.replace('long-1yearNeo_', 'long_')
        else:
            featKey = featKey.replace('2year_', '')
        featList = featList + [featKey]

    # print featList

    print "%d Features sorted by their %s:" % (len(featList), sf)

    return featList


def ShowFinalFeats(featureCOEForSCOREList, threshold, showFeat, sf):
    # print featureCOEForSCOREList
    threshold = 0
    featureCOEForSCOREList = [f for f in featureCOEForSCOREList if f[0] > threshold]
    featureCOEForSCOREList = map(lambda f: (f[1], f[0]), featureCOEForSCOREList)

    featureCOEForSCOREListDic = dict(featureCOEForSCOREList)

    newdic = {}

    for feat in featureCOEForSCOREListDic.keys():
        # print feat
        newdic[feat] = np.mean(featureCOEForSCOREListDic[feat])

    newdic = dict((v, k) for k, v in newdic.iteritems())

    featureCOEForSCOREList = list(newdic.items())

    print "%d Features sorted by their %s:" % (len(featureCOEForSCOREList), sf)
    sortedFeatList = sorted(featureCOEForSCOREList, reverse=True)

    featNamesLong = []
    featNamesCros = []
    featScoresLong = []
    featScoresCros = []

    for Feat in sortedFeatList:
        print repr(Feat[1]), ',', repr(Feat[0])
        if 'long' in Feat[1]:
            new_Feat = Feat[1].replace('long-1yearNeo_', '')
            featScoresLong.append(Feat[0])
            featNamesLong.append(new_Feat)
        else:
            new_Feat = Feat[1].replace('2year_', '')
            featScoresCros.append(Feat[0])
            featNamesCros.append(new_Feat)

    fig = plt.figure()

    width = .35
    indLong = np.arange(len(featScoresLong))
    indCros = np.arange(len(featScoresCros))

    plt.bar(indLong, featScoresLong, width=width, color='b', align='center', alpha=0.5)
    plt.xticks(indLong + width / 2, featNamesLong, rotation=90, fontsize='small')
    plt.ylabel('Coefficients')
    plt.xlabel("Variables")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/Diagrams/features_Long.eps", format='eps', dpi=1000)
    plt.show()

    plt.bar(indCros, featScoresCros, width=width, color='r', align='center', alpha=0.5)
    plt.xticks(indCros + width / 2, featNamesCros, rotation=90, fontsize='small')
    plt.ylabel('Coefficients')
    plt.xlabel("Variables")
    plt.grid(True)
    plt.tight_layout()
    # fig.autofmt_xdate()
    plt.savefig("results/Diagrams/features_Cros.eps", format='eps', dpi=1000)
    plt.show()


def preprocess_Data(unlabel, pre_process, X, X_balanced, X_MinusX_balanced, UnLabeledData, UnLabeledData_balanced):

    if pre_process == 'norm':
        # Normalization rescale the data between -1 and 1
        # norm can be 'l1','l2','max'
        # X=preprocessing.normalize(X,norm='l2')

        X_scaler = preprocessing.MinMaxScaler()
        X = X_scaler.fit_transform(X)
        X_balanced = X_scaler.fit_transform(X_balanced)
        X_MinusX_balanced = X_scaler.fit_transform(X_MinusX_balanced)

        if unlabel:
            # UnLabeledData=preprocessing.normalize(UnLabeledData,norm='l2')
            UnLabeledData_scaler = preprocessing.MinMaxScaler()
            UnLabeledData = UnLabeledData_scaler.fit_transform(UnLabeledData)
            UnLabeledData_balanced = UnLabeledData_scaler.fit_transform(UnLabeledData_balanced)
    else:
        # Standardization
        # Shift the distribution to have mean of zero and standard devisation of 1 (unit Variance)
        X = preprocessing.scale(X)
        X_balanced = preprocessing.scale(X_balanced)
        X_MinusX_balanced = preprocessing.scale(X_MinusX_balanced)

        if unlabel:
            UnLabeledData = preprocessing.scale(UnLabeledData)
            UnLabeledData_balanced = preprocessing.scale(UnLabeledData_balanced)

    return X, X_balanced, X_MinusX_balanced, UnLabeledData, UnLabeledData_balanced


def DimensionReduction(feature_names, method, X_test, X_train, y_train, dimension, UnlabeledData, dataflag, showFeat):
    threshold = 0
    featureCOEForSCOREList = []
    print "number of dimension = %d" % dimension
    if dataflag == 'all' and UnlabeledData != []:
        Train_and_UnLabeled = np.vstack((X_train, UnlabeledData))
    else:
        Train_and_UnLabeled = X_train
    if (method == 'na'):
        threshold = 0
        pass
    if (method == 'pca'):
        # Principle Component Analysis
        # The PCA(0 without indicating the number of components will chose: n_components=min(n_features,n_samples)
        # pca = PCA()
        # Xy=np.concatenate((X,y),axis=1)
        # pca.fit(X_train)
        # X_train = pca.transform(X_train)
        # To be able to chose number of componets automaticly Minca's method we need n_features<=n_samples
        # pca = PCA(n_components='mle')
        pca = PCA(n_components=dimension)
        pca.fit(Train_and_UnLabeled)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        UnlabeledData = pca.transform(UnlabeledData)

    if (method == 'lda'):
        # LDA dimention reduction number of diemnsion can be maximum (number of classes-1)
        lda = LinearDiscriminantAnalysis(n_components=dimension)
        lda.fit(X_train, y_train)
        X_train = lda.transform(X_train)
        X_test = lda.transform(X_test)
        UnlabeledData = lda.transform(UnlabeledData)
    if (method == 'pls'):
        pls = PLSRegression(n_components=dimension)
        pls.fit(X_train, y_train)
        X_train = pls.transform(X_train)
        X_test = pls.transform(X_test)
        UnlabeledData = pls.transform(UnlabeledData)
    if (method == 'sparsepca'):
        # Sparse Principle Component Analysis
        spca = SparsePCA(n_components=dimension)
        spca.fit(Train_and_UnLabeled)
        X_train = spca.transform(X_train)
        X_test = spca.transform(X_test)
        UnlabeledData = spca.transform(UnlabeledData)
    if (method == 'ica'):
        # Indipendent Component Analysis
        ica = FastICA(n_components=dimension)
        ica.fit(Train_and_UnLabeled)
        X_train = ica.transform(X_train)
        X_test = ica.transform(X_test)
        UnlabeledData = ica.transform(UnlabeledData)
    if (method == 'nmf'):
        # Non-Negative Matrix Factorization
        ####NMF using spams
        # (U,V) = spams.nmf( np.transpose(X_train), return_lasso = True, K = dimension,iter=-5 )
        # mapped_X = X.dot(U)
        # X=mapped_X
        nmf = NMF(n_components=dimension)
        nmf.fit(Train_and_UnLabeled)
        X_train = nmf.transform(X_train)
        X_test = nmf.transform(X_test)
        UnlabeledData = nmf.transform(UnlabeledData)
    if (method == 'ridge'):
        # Reconstruction with L2 (Ridge) penalization
        # alpha defines the amount of shrinkage
        threshold = 0
        alpha_range = np.logspace(-5, 3, 30)
        ridge = linear_model.RidgeCV(alphas=alpha_range)
        ridge.fit(X_train, y_train)
        print ridge.alpha_
        clf = linear_model.Ridge(alpha=ridge.alpha_)
        clf.fit(X_train, y_train)
        masked_coef = np.ma.masked_greater(clf.coef_, 0).mask

        featureCOEForSCOREList = ShowSelectedFeat('coef', clf.coef_, feature_names, threshold, showFeat)

        # print clf.coef_
        # clf.coef_[clf.coef_>0] = 1
        # clf.coef_[clf.coef_<=0] = 0
        X_train = X_train[:, masked_coef]
        X_test = X_test[:, masked_coef]
        UnlabeledData = UnlabeledData[:, masked_coef]
    if (method == 'lasso'):
        # Reconstruction with L1 (Lasso) penalization
        # the best value of alpha was determined using cross validation with LassoCV
        threshold = 0
        alpha_range = np.logspace(-5, 1, 30)
        lasso = linear_model.LassoCV(alphas=alpha_range, positive=True)
        lasso.fit(X_train, y_train)
        print 'Lasso Alpha is = ', lasso.alpha_
        clf = linear_model.Lasso(alpha=lasso.alpha_, positive=True,random_state=7)
        clf.fit(X_train, y_train)
        masked_coef = np.ma.masked_greater(clf.coef_, 0).mask
        # print clf.coef_

        featureCOEForSCOREList = ShowSelectedFeat('coef', clf.coef_, feature_names, threshold, showFeat)

        X_train = X_train[:, masked_coef]
        X_test = X_test[:, masked_coef]
        UnlabeledData = UnlabeledData[:, masked_coef]
    if (method == 'tree_based'):
        threshold = 0
        clf = ExtraTreesClassifier()
        clf = clf.fit(X_train, y_train)
        # print(clf.feature_importances_)
        featureCOEForSCOREList = ShowSelectedFeat('Tree based feature importances', clf.feature_importances_,
                                                  feature_names, threshold, showFeat)

        model = SelectFromModel(clf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)
    if (method == 'randomForest'):
        threshold = 0
        rf = RandomForestRegressor()
        rf = rf.fit(X_train, y_train)

        featureCOEForSCOREList = ShowSelectedFeat('RandomForest feature importances', rf.feature_importances_,
                                                  feature_names, threshold, showFeat)

        model = SelectFromModel(rf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)
    if (method == 'elasticNet'):
        # ElasticNet
        threshold = 0
        alpha_range = np.logspace(-5, 0, 30)
        l1_ratio_range = [.1, .5, .7, .9, .95, .99, 1]
        elnet = linear_model.ElasticNetCV(alphas=alpha_range, l1_ratio=l1_ratio_range, positive=True)
        elnet.fit(X_train, y_train)
        print elnet.alpha_, elnet.l1_ratio_
        # clf = linear_model.ElasticNet(alpha=0.5,l1_ratio=1,positive=True)
        clf = linear_model.ElasticNet(alpha=elnet.alpha_, l1_ratio=elnet.l1_ratio_, positive=True)
        clf.fit(X_train, y_train)
        masked_coef = np.ma.masked_greater(clf.coef_, 0).mask
        # print clf.coef_
        featureCOEForSCOREList = ShowSelectedFeat('coef', clf.coef_, feature_names, threshold, showFeat)

        X_train = X_train[:, masked_coef]
        X_test = X_test[:, masked_coef]
        UnlabeledData = UnlabeledData[:, masked_coef]
    if (method == 'stabilitySel_logistic'):
        # Randomized Logistic Regression Know as Stability Selection with Classification properties(output class labels)

        # rlr = linear_model.RandomizedLogisticRegression()
        # grid=Parameter_tunning('rlr',rlr,X_train,y_train,X_test,[],0,0,'r2')
        # print grid_clf.best_params_
        # rlr = linear_model.RandomizedLogisticRegression(scaling=grid_clf.best_params_['scaling'],C=grid_clf.best_params_['C'])
        threshold = 0
        rlr = linear_model.RandomizedLogisticRegression(selection_threshold=threshold)
        rlr.fit(X_train, y_train)

        featureCOEForSCOREList = ShowSelectedFeat('score', rlr.scores_, feature_names, threshold, showFeat)

        X_train = rlr.transform(X_train)
        X_test = rlr.transform(X_test)
        UnlabeledData = rlr.transform(UnlabeledData)
    if (method == 'stabilitySel_lasso'):
        # Randomized Lasso Know as Stability Selection with regression properties(output continues values that is why it needs threshold)
        # since I chose the threshold = 0, I also normalized the parameters between -1 to 1
        threshold = 0
        """alpha_range=np.logspace(-5,1,30)

        rl = linear_model.RandomizedLasso(selection_threshold=threshold)

        grid=Parameter_tunning('rl',rl,X_train,y_train,X_test,[],0,0,'r2')
        clf=grid.best_estimator_

        model = SelectFromModel(clf,prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)"""

        rl = linear_model.RandomizedLasso(selection_threshold=threshold,random_state=2)
        rl.fit(X_train, y_train)
        # print rl.scores_
        featureCOEForSCOREList = ShowSelectedFeat('score', rl.scores_, feature_names, threshold, showFeat)
        print "X_train", X_train.shape
        X_train = rl.transform(X_train)
        print "X_train", X_test.shape
        X_test = rl.transform(X_test)
        print "Unlabeled", UnlabeledData.shape
        UnlabeledData = rl.transform(UnlabeledData)
    if (method == 'featSel_SVM'):
        # 'LINEAR SVM'
        linear_svm = svm.LinearSVC(penalty="l1", dual=False)
        C_range = np.logspace(-1, 1, 3)
        param_grid = {'C': C_range}
        grid = GridSearchCV(linear_svm, param_grid, verbose=1)
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_

        model = SelectFromModel(clf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)
        print model.threshold_
    if (method == 'featSel_ELASTIC'):
        # 'ELASTIC NET'
        threshold = 0
        alpha_range = np.logspace(-5, 0, 30)
        l1_ratio_range = [.1, .5, .7, .9, .95, .99, 1]
        elnet = linear_model.ElasticNetCV(alphas=alpha_range, l1_ratio=l1_ratio_range, positive=True)
        elnet.fit(X_train, y_train)
        print elnet.alpha_, elnet.l1_ratio_
        clf = linear_model.ElasticNet(alpha=elnet.alpha_, l1_ratio=elnet.l1_ratio_, positive=True)
        clf = clf.fit(X_train, y_train)
        # print clf.coef_
        featureCOEForSCOREList = ShowSelectedFeat('coef', clf.coef_, feature_names, threshold, showFeat)

        model = SelectFromModel(clf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)
        print model.threshold_
    if (method == 'featSel_LASSO'):
        # 'LASSO'
        threshold = 0
        alpha_range = np.logspace(-5, 0, 30)
        lasso = linear_model.LassoCV(alphas=alpha_range, positive=True)
        lasso.fit(X_train, y_train)
        print lasso.alpha_
        clf = linear_model.Lasso(alpha=lasso.alpha_, positive=True)
        clf = clf.fit(X_train, y_train)
        # print clf.coef_
        featureCOEForSCOREList = ShowSelectedFeat('coef', clf.coef_, feature_names, threshold, showFeat)

        model = SelectFromModel(clf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)
        print model.threshold_
    if (method == 'featSel_RIDGE'):
        # 'RIDGE'
        threshold = 0
        alphas = np.logspace(-1, 1, 30)
        ridge = linear_model.RidgeCV(alphas=alphas)
        ridge.fit(X_train, y_train)
        clf = linear_model.Ridge(alpha=ridge.alpha_)
        clf = clf.fit(X_train, y_train)
        # print clf.coef_
        featureCOEForSCOREList = ShowSelectedFeat('coef', clf.coef_, feature_names, threshold, showFeat)

        model = SelectFromModel(clf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
        UnlabeledData = model.transform(UnlabeledData)
        print model.threshold_

    if (method == 'nnsc'):
        # non-negative sparse coding
        (U, V) = spams.nnsc(np.transpose(Train_and_UnLabeled), return_lasso=True, K=dimension)
        X_train = X_train.dot(U)
        X_test = X_test.dot(U)
        UnlabeledData = UnlabeledData.dot(U)
    return X_train, X_test, UnlabeledData, featureCOEForSCOREList, threshold


def Parameter_tunning(method, model, X_train, y_train, X_test, y_test, ver, print_folds, scoring):
    if print_folds:
        print "Perfoming hyper-parameters Tunning ..."

    alpha_range = np.logspace(-5, 0, 30)
    # C_range = np.logspace(-3,5, 9)
    C_range = np.logspace(-1, 1, 3)
    gamma_range = np.logspace(-5, 3, 9)
    k_range = range(1, 31)
    # degree_range=range(1,6)
    degree_range = range(2, 4)
    weight_options = ['uniform', 'distance']

    if (method == 'knn'):
        param_grid = dict(n_neighbors=k_range, weights=weight_options)
    if (method == 'linear_svm'):
        param_grid = {'C': C_range}
    if (method == 'rbf_svm'):
        param_grid = {'C': C_range, 'gamma': gamma_range}
    if (method == 'poly_svm'):
        param_grid = {'C': C_range, 'degree': degree_range}
    if (method == 'lr'):
        param_grid = {'C': C_range}
    if (method == 'dt'):
        param_grid = {  # "criterion": ["gini", "entropy"],
            # "min_samples_split": [2, 10, 20],
            # "max_depth": [None, 2, 5, 10],
            # "min_samples_leaf": [1, 5, 10],
            "max_leaf_nodes": [None, 5, 10, 20]
        }
    if (method == 'rfc'):
        """param_grid = {#"n_estimators": [200,700],
                          #"max_features": ['auto', 'sqrt', 'log2'],
                          "max_features": [1, 3, 10]	              
                      #"bootstrap": [True, False],              		      
                      }"""

        param_grid = {  # "criterion": ["gini", "entropy"],
            # "min_samples_split": [2, 10, 20],
            # "max_depth": [None, 2, 5, 10],
            # "min_samples_leaf": [1, 5, 10],
            "max_leaf_nodes": [None, 5, 10, 20]
        }
    if (method == 'rlr'):
        param_grid = {'C': C_range, 'scaling': alpha_range}

    if (method == 'rl'):
        param_grid = {'alpha': alpha_range, 'scaling': alpha_range}
    if (method == 'mlp'):
        param_grid = {
            'alpha': 10.0 ** -np.arange(1, 7)
        }

    # verbose : integer, Controls the verbosity: the higher, the more messages, default is 0.
    grid = GridSearchCV(model, param_grid, verbose=ver, scoring=scoring)
    grid.fit(X_train, y_train)
    if print_folds:
        print "Result of tunning on Training Data is:"
        print "Best %s = %f, best parametrs are:" % (scoring, grid.best_score_), grid.best_params_

    return grid


def Read_Data(age, dataflag, unlabel, fix_corrupt, readFrom):
    if age == 'long':
        Data_path = './%s_files/%scomp_Brain_cog%s.csv' % (age, age, readFrom)
        DataSub_path = './%s_files/%scomp_BrainSubset%s.csv' % (age, age, readFrom)
        UnLabeledDataSub_path = './%s_files/%scomp_NoLabelBrainSubset.csv' % (age, age)
        UncompData_path = './%s_files/%scomp_BrainUncomp%s.csv' % (age, age, readFrom)
        UnLabeledData_path = './%s_files/%scomp_BrainData.csv' % (age, age)
        LTcompData_path = './%s_files/%scomp_BrainLTcomp_concat.csv' % (age, age)

        delta_path = './%s_files/%sdelta_Brain_cog%s.csv' % (age, age, readFrom)
        UnLabled_delta_path = './%s_files/%sdelta_BrainData.csv' % (age, age)

        deltasub_path = './%s_files/%sdelta_BrainSub%s.csv' % (age, age, readFrom)
        UnLabled_deltasub_path = './%s_files/%sdelta_BrainDataSub.csv' % (age, age)

        deltasub2y1y_path = './%s_files/%sdelta_2year1year_BrainSub%s.csv' % (age, age, readFrom)
        UnLabled_deltasub2y1y_path = './%s_files/%sdelta_2year1year_BrainDataSub.csv' % (age, age)

        deltasub1yNeo_path = './%s_files/%sdelta_1yearNeo_BrainSub%s.csv' % (age, age, readFrom)
        UnLabled_deltasub1yNeo_path = './%s_files/%sdelta_1yearNeo_BrainDataSub.csv' % (age, age)

        Fiber_path = './%s_files/%scomp_Fiber%s.csv' % (age, age, readFrom)
        FiberEnigma_path = './%s_files/%scomp_FiberEnigma%s.csv' % (age, age, readFrom)
    else:
        Data_path = './%s_files/%s_Brain_cog%s.csv' % (age, age, readFrom)
        UnLabeledDataSub_path = './%s_files/%s_NoLabelBrainSubsetFeats.csv' % (age, age)
        DataSub_path = './%s_files/%s_BrainSubsetFeats%s.csv' % (age, age, readFrom)
        Fiber_path = './%s_files/%s_Fiber%s.csv' % (age, age, readFrom)
        FiberEnigma_path = './%s_files/%s_FiberEnigma%s.csv' % (age, age, readFrom)
        UncompData_path = './%s_files/%s_BrainUncomp%s.csv' % (age, age, readFrom)
        UnLabeledData_path = './%s_files/%s_BrainData.csv' % (age, age)
        LTcompData_path = './%s_files/%s_BrainLTcomp.csv' % (age, age)

    if dataflag == 'delta':
        DataFpath = delta_path
        DataF = pd.read_csv(delta_path)
        UnLabeledData_path = UnLabled_delta_path
    if dataflag == 'deltaSub':
        DataFpath = deltasub_path
        DataF = pd.read_csv(deltasub_path)
        UnLabeledData_path = UnLabled_deltasub_path
    if dataflag == 'deltaSub2y1y':
        DataFpath = deltasub2y1y_path
        DataF = pd.read_csv(deltasub2y1y_path)
        UnLabeledData_path = UnLabled_deltasub2y1y_path
    if dataflag == 'deltaSub1yNeo':
        DataFpath = deltasub1yNeo_path
        DataF = pd.read_csv(deltasub1yNeo_path)
        UnLabeledData_path = UnLabled_deltasub1yNeo_path
    if dataflag == 'ZeroComp':
        unlabel = False
        DataFpath = UncompData_path
        DataF = pd.read_csv(UncompData_path)
        DataF = DataF.fillna(0)
    if dataflag == 'LTComp':
        unlabel = False
        DataFpath = LTcompData_path
        DataF = pd.read_csv(LTcompData_path)
        DataF = DataF.fillna(0)
    if dataflag == 'all':
        DataFpath = Data_path
        DataF = pd.read_csv(Data_path)
    if dataflag == 'sub':
        DataFpath = DataSub_path
        UnLabeledData_path = UnLabeledDataSub_path
        DataF = pd.read_csv(DataSub_path)
    if dataflag == 'fib':
        if age == 'long':
            UnLabeledData_path = './%s_files/%scomp_UnFiber.csv' % (age, age)
        else:
            UnLabeledData_path = './%s_files/%s_UnFiber.csv' % (age, age)
        DataFpath = Fiber_path
        DataF = pd.read_csv(Fiber_path)
    if dataflag == 'fibEnig':
        if age == 'long':
            UnLabeledData_path = './%s_files/%scomp_UnFiberEnigma.csv' % (age, age)
        else:
            UnLabeledData_path = './%s_files/%s_UnFiberEnigma.csv' % (age, age)
        DataFpath = FiberEnigma_path
        DataF = pd.read_csv(FiberEnigma_path)

    DataF = DataF.dropna()
    if unlabel:
        UnLabeledData = pd.read_csv(UnLabeledData_path)
        print "new add: UnLabeledData_path=", UnLabeledData_path
        print "-------------------------------------UnlabeledData Dimensions:", UnLabeledData.shape
        # This line will remove the DataF from UnLabeledData to perevnet from test data used for our dimension reduction
        UnLabeledData = UnLabeledData.dropna()
        UnLabeledData = UnLabeledData.loc[UnLabeledData['subjectID'].isin(DataF['subjectID']) == False]

        if fix_corrupt:
            UnLabeledData = find_corrupted_digits(UnLabeledData)
            print UnLabeledData_path
            UnLabeledData.to_csv(UnLabeledData_path, index=False)
    if fix_corrupt:
        DataF = find_corrupted_digits(DataF)
        print DataFpath
        DataF.to_csv(DataFpath, index=False)
    # print DataF.columns
    ######## WRITE A CODE TO CHANGE THE FEATURE NAMES
    return DataF, UnLabeledData


def return_DataFrames(age, dataflag, corr, mergeWith, gestAge):
    if mergeWith == '':

        [DataF, UnLabeledData] = Read_Data(age, dataflag, unlabel, fix_corrupt, '')
        print "Data dimension =", DataF.shape

        [DataF_balanced, UnLabeledData_balanced] = Read_Data(age, dataflag, unlabel, fix_corrupt, '_balanced')
        print "Data Balanced dimension =", DataF_balanced.shape
    else:
        [DataF1, UnLabeledData1] = Read_Data(age, dataflag, unlabel, fix_corrupt, '')
        [DataF1_balanced, UnLabeledData1_balanced] = Read_Data(age, dataflag, unlabel, fix_corrupt, '_balanced')

        DataF1 = change_col_name(DataF1, age + '-')
        DataF1_balanced = change_col_name(DataF1_balanced, age + '-')

        UnLabeledData1 = change_col_name(UnLabeledData1, age + '_')
        UnLabeledData1_balanced = change_col_name(UnLabeledData1_balanced, age + '_')

        [age2, dataflag2] = mergeWith.split('_')

        [DataF2, UnLabeledData2] = Read_Data(age2, dataflag2, unlabel, fix_corrupt, '')
        [DataF2_balanced, UnLabeledData2_balanced] = Read_Data(age2, dataflag2, unlabel, fix_corrupt, '_balanced')

        DataF2 = change_col_name(DataF2, age2 + '_')
        UnLabeledData2 = change_col_name(UnLabeledData2, age2 + '_')
        DataF2_balanced = change_col_name(DataF2_balanced, age2 + '_')
        UnLabeledData2_balanced = change_col_name(UnLabeledData2_balanced, age2 + '_')

        print age + '-', age2 + '_'

        DataF = DataF1.merge(DataF2, on='subjectID', how='inner')
        DataF['Label'] = DataF[age + '-Label']
        DataF = DataF.drop([age + '-Label', age2 + '_Label'], axis=1)
        UnLabeledData = UnLabeledData1.merge(UnLabeledData2, on='subjectID', how='inner')
        print "Data dimension =", DataF.shape

        DataF_balanced = DataF1_balanced.merge(DataF2_balanced, on='subjectID', how='inner')
        DataF_balanced['Label'] = DataF_balanced[age + '-Label']
        DataF_balanced = DataF_balanced.drop([age + '-Label', age2 + '_Label'], axis=1)
        UnLabeledData_balanced = UnLabeledData1_balanced.merge(UnLabeledData2_balanced, on='subjectID', how='inner')
        print "Data Balanced dimension =", DataF_balanced.shape
    if gestAge == True:
        Subject_gestAge = pd.read_csv('ConteTwinDemog.csv')
        Subject_gestAge = Subject_gestAge[['subjectID', 'GestAgeBirth']]

        UnLabeledData = UnLabeledData.merge(Subject_gestAge, on='subjectID', how='inner')
        DataF = DataF.merge(Subject_gestAge, on='subjectID', how='inner')

        UnLabeledData_balanced = UnLabeledData_balanced.merge(Subject_gestAge, on='subjectID', how='inner')
        DataF_balanced = DataF_balanced.merge(Subject_gestAge, on='subjectID', how='inner')
    if corr == True:
        visualize_corelation(DataF)
        visualize_corelation(DataF_balanced)

    return DataF, DataF_balanced, UnLabeledData, UnLabeledData_balanced


def Data_load_normalize(age, dataflag, dimred_method, unlabel, fix_corrupt, corr, pre_process, stat, mergeWith, gestAge, train_balanced):
    if dimred_method in ['stabilitySel_lasso', 'lasso', 'elasticNet', 'nnsc', 'nmf', 'featSel_LASSO', 'featSel_ELASTIC', 'featSel_RIDGE']:
        pre_process = 'norm'

    [DataF, DataF_balanced, UnLabeledData, UnLabeledData_balanced] = return_DataFrames(age, dataflag, corr, mergeWith, gestAge)

    DataF_MinusData_balanced = DataF.loc[DataF['subjectID'].isin(DataF_balanced['subjectID']) == False]
    DataF_MinusData_balanced = DataF_MinusData_balanced.reset_index(drop=True)

    column_header = 'Label'

    subjects = DataF['subjectID']
    DataF = DataF.ix[:, DataF.columns != 'subjectID']


    feature_names = [name for name in DataF.columns if name != column_header]
    labels = DataF[column_header]
    Data = DataF.ix[:, DataF.columns != column_header]

    subjects_balanced = DataF_balanced['subjectID']
    DataF_balanced = DataF_balanced.ix[:, DataF_balanced.columns != 'subjectID']


    labels_balanced = DataF_balanced[column_header]
    Data_balanced = DataF_balanced.ix[:, DataF_balanced.columns != column_header]

    subjects_Minus = DataF_MinusData_balanced['subjectID']

    DataF_MinusData_balanced = DataF_MinusData_balanced.ix[:, DataF_MinusData_balanced.columns != 'subjectID']
    labels_Minus = DataF_MinusData_balanced[column_header]
    Data_MinusData_balanced = DataF_MinusData_balanced.ix[:, DataF_MinusData_balanced.columns != column_header]

    At_Risk_no = sum(labels)
    Typical_no = len(labels) - sum(labels)
    print "Typical number=%d, At Risk number=%d" % (Typical_no, At_Risk_no)
    label_names = ['Typical', 'At_Risk']

    if unlabel:
        UnLabeledData = UnLabeledData.ix[:, UnLabeledData.columns != 'subjectID']
        UnLabeledData = np.array(UnLabeledData)
        UnLabeledData_balanced = UnLabeledData_balanced.ix[:, UnLabeledData_balanced.columns != 'subjectID']
        UnLabeledData_balanced = np.array(UnLabeledData_balanced)

    X, y = np.array((Data.values)), np.array(labels)
    X_balanced, y_balanced = np.array((Data_balanced.values)), np.array(labels_balanced)
    X_MinusX_balanced, y_Minusy_balanced = np.array((Data_MinusData_balanced.values)), np.array(labels_Minus)

    print "X=", X.shape
    print "X_balanced=", X_balanced.shape
    print "X_MinusX_balanced=", X_MinusX_balanced.shape
    if unlabel:
        print 'Unlabled after removing X=', UnLabeledData.shape
        print 'Unlabled after removing X_balanced=', UnLabeledData_balanced.shape

    [X, X_balanced, X_MinusX_balanced, UnLabeledData, UnLabeledData_balanced] = preprocess_Data(unlabel, pre_process, X,
                                                                                                X_balanced,
                                                                                                X_MinusX_balanced,
                                                                                                UnLabeledData,
                                                                                                UnLabeledData_balanced)

    if train_balanced:
        X_data = X_balanced
        y_data = y_balanced
        #	UnlabeledData after removing unblanced Data
        #	UnLabeledData_balanced after removing the balanced Data so we use this one on train since our train data is balanced
        UnlabeledData = UnLabeledData_balanced
        subjects_data = subjects_balanced
    else:
        X_data = X
        y_data = y
        subjects_data = subjects

    return subjects_data, subjects_Minus, feature_names, X_data, y_data, X_MinusX_balanced, y_Minusy_balanced, UnlabeledData


def print_confidenceScore(classifier_name, expected, predicted, confidence_scores):
    print "\n", classifier_name, ":"
    print '(expected , predicted , confidence_score)\n'
    for p, e, cs in zip(expected, predicted, confidence_scores):
        print  '(', p, ',', e, ',', cs, ')\n'


def multiplier_func(expected):
    mul_cs = []
    for x in expected:
        if x == 1:
            mul_cs = mul_cs + [1]
        else:
            mul_cs = mul_cs + [-1]
    return mul_cs


def FindConfScore_DecFunc(X_test, expected, predicted, classifier, classifier_name, print_confidence):
    mul_cs = multiplier_func(expected)
    uncorrectedSign_confidence_scores = classifier.decision_function(X_test)
    confidence_scores = uncorrectedSign_confidence_scores * mul_cs
    if print_confidence:
        print_confidenceScore(classifier_name, expected, predicted, confidence_scores)

    return confidence_scores


def featureFusion(Pred_StructMRI, Pred_DTI,Pred_StructMRI_noGes):
    Pred_StructMRI = Pred_StructMRI.rename(columns={'Predicted_Label': 'Pred_StructMRI_PL'})
    Pred_StructMRI = Pred_StructMRI.rename(columns={'confidence_score': 'Pred_StructMRI_CS'})

    Pred_StructMRI_noGes = Pred_StructMRI_noGes.rename(columns={'Predicted_Label': 'Pred_StructMRI_noGes_PL'})
    Pred_StructMRI_noGes = Pred_StructMRI_noGes.rename(columns={'confidence_score': 'Pred_StructMRI_noGes_CS'})

    Pred_DTI = Pred_DTI.rename(columns={'Predicted_Label': 'Pred_DTI_PL'})
    Pred_DTI = Pred_DTI.rename(columns={'confidence_score': 'Pred_DTI_CS'})

    merge_DTI_StructMRI = Pred_StructMRI.merge(Pred_DTI, on='subjectID', how='inner')

    Real_label_list = []
    for x, y in zip(merge_DTI_StructMRI['Real_Label_x'], merge_DTI_StructMRI['Real_Label_y']):
        if np.isnan(x):
            Real_label_list.append(y)
        else:
            Real_label_list.append(x)

    merge_DTI_StructMRI.loc[
        (merge_DTI_StructMRI['Pred_StructMRI_CS'] >= merge_DTI_StructMRI['Pred_DTI_CS']), 'Predict_Label'] = \
    merge_DTI_StructMRI['Pred_StructMRI_PL']
    merge_DTI_StructMRI.loc[
        ~(merge_DTI_StructMRI['Pred_StructMRI_CS'] >= merge_DTI_StructMRI['Pred_DTI_CS']), 'Predict_Label'] = \
    merge_DTI_StructMRI['Pred_DTI_PL']

    merge_DTI_StructMRI['Real_Label'] = Real_label_list

    merge_DTI_StructMRI = merge_DTI_StructMRI.drop(['Real_Label_x'], axis=1)
    merge_DTI_StructMRI = merge_DTI_StructMRI.drop(['Real_Label_y'], axis=1)

    print ' Number of subjects: Final Merge for DTI&StructMRI = %d, StructMRI = %d, DTI = %d' % (
    len(merge_DTI_StructMRI), len(Pred_StructMRI), len(Pred_DTI))
    print merge_DTI_StructMRI
    [accuracy, sensitivity, specificity, ppv, npv, cm] = print_result(merge_DTI_StructMRI['Real_Label'],
                                                                      merge_DTI_StructMRI['Predict_Label'])

    def show_accuracy(y_test, y_predict):

        from sklearn.metrics import average_precision_score
        from sklearn.metrics import precision_recall_fscore_support
        # average_precision = average_precision_score(y_test, y_predict)

        # print('Average precision-recall score: {0:0.2f}'.format(
        #      average_precision))



        average_precision = average_precision_score(y_test, y_predict)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        precision, recall, _ = precision_recall_curve(y_test, y_predict)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
            average_precision))

        [per, rec, f1, NoImp] = metrics.precision_recall_fscore_support(y_test, y_predict, average='macro')
        acc = metrics.accuracy_score(y_test, y_predict)
        print "Accuracy = %f percision = %f recall= %f F1_score= %f" % (acc, per, rec, f1)

    ##############################
    merge_DTI_StructMRI_noGes = Pred_StructMRI_noGes.merge(Pred_DTI, on='subjectID', how='inner')

    Real_label_list = []
    for x, y in zip(merge_DTI_StructMRI_noGes['Real_Label_x'], merge_DTI_StructMRI_noGes['Real_Label_y']):
        if np.isnan(x):
            Real_label_list.append(y)
        else:
            Real_label_list.append(x)

    merge_DTI_StructMRI_noGes.loc[
        (merge_DTI_StructMRI_noGes['Pred_StructMRI_noGes_CS'] >= merge_DTI_StructMRI_noGes['Pred_DTI_CS']), 'Predict_Label'] = \
        merge_DTI_StructMRI_noGes['Pred_StructMRI_noGes_PL']
    merge_DTI_StructMRI_noGes.loc[
        ~(merge_DTI_StructMRI_noGes['Pred_StructMRI_noGes_CS'] >= merge_DTI_StructMRI_noGes['Pred_DTI_CS']), 'Predict_Label'] = \
        merge_DTI_StructMRI_noGes['Pred_DTI_PL']

    merge_DTI_StructMRI_noGes['Real_Label'] = Real_label_list

    merge_DTI_StructMRI_noGes = merge_DTI_StructMRI_noGes.drop(['Real_Label_x'], axis=1)
    merge_DTI_StructMRI_noGes = merge_DTI_StructMRI_noGes.drop(['Real_Label_y'], axis=1)

    """print ' Number of subjects: Final Merge for DTI&StructMRI = %d, StructMRI = %d, DTI = %d' % (
        len(merge_DTI_StructMRI), len(Pred_StructMRI), len(Pred_DTI))
    print merge_DTI_StructMRI"""
    [accuracy_noGes, sensitivity_noGes, specificity_noGes, ppv_noGes, npv_noGes, cm_noGes] = print_result(merge_DTI_StructMRI_noGes['Real_Label'],
                                                                      merge_DTI_StructMRI_noGes['Predict_Label'])


    ################################
    labels = ['Typical', 'At_Risk']

    [str_accuracy, str_sensitivity, str_specificity, str_ppv, str_npv, str_cm] = print_result(Pred_StructMRI_noGes['Real_Label'],Pred_StructMRI_noGes['Pred_StructMRI_noGes_PL'])
    show_accuracy(Pred_StructMRI_noGes['Real_Label'],Pred_StructMRI_noGes['Pred_StructMRI_noGes_PL'])
    [dti_accuracy, dti_sensitivity, dti_specificity, dti_ppv, dti_npv, dti_cm] = print_result(Pred_DTI['Real_Label'],Pred_DTI['Pred_DTI_PL'])
    show_accuracy(Pred_DTI['Real_Label'],Pred_DTI['Pred_DTI_PL'])
    n_groups = 5
    Acc_StructMRI = np.array([str_accuracy, str_sensitivity, str_specificity, str_ppv, str_npv])
    Acc_DTI = np.array([dti_accuracy, dti_sensitivity, dti_specificity, dti_ppv, dti_npv])
    ACC_FF = np.array([accuracy_noGes, sensitivity_noGes, specificity_noGes, ppv_noGes, npv_noGes])
    Acc_FF_G = np.array([accuracy, sensitivity, specificity, ppv, npv])
    # plot_confusion_matrix(cm, classes=labels,normalize=False,title='Confusion matrix, without normalization')
    # plt.savefig('result_cm.eps', format='eps', dpi=1000)

    show_confusion_matrix(cm, class_labels=labels)

    show_bar_diagram(n_groups, Acc_StructMRI, Acc_DTI,ACC_FF,Acc_FF_G)
    # data to plot


    show_Results('StructralMRI:', str_accuracy, str_sensitivity, str_specificity, str_ppv, str_npv, str_cm, 1)
    [str_accuracy_Ges, str_sensitivity_Ges, str_specificity_Ges, str_ppv_Ges, str_npv_Ges, str_cm_Ges] = print_result(Pred_StructMRI['Real_Label'], Pred_StructMRI['Pred_StructMRI_PL'])
    show_Results('StructralMRI + Gestational Age:', str_accuracy_Ges, str_sensitivity_Ges, str_specificity_Ges, str_ppv_Ges, str_npv_Ges, str_cm_Ges, 1)
    show_Results('DTI:', dti_accuracy, dti_sensitivity, dti_specificity, dti_ppv, dti_npv,dti_cm, 1)
    show_Results('Structral + DTI:', accuracy_noGes, sensitivity_noGes, specificity_noGes, ppv_noGes, npv_noGes, cm_noGes, 1)
    show_Results('Structral + DTI + Gestational Age:', accuracy, sensitivity, specificity, ppv, npv,cm, 1)

def classification(binary, age, no_folds, no_dims, dimred_method, stat, ver, print_folds, dataflag,
                   pre_process, unlabel, synthetic, corr, fix_corrupt, showFeat, scoring, mergeWith, gestAge,
                   train_balanced, test_balanced, print_confidence, cls_method):
    if cls_method == 'semi':
        unlabel = True
    # mlp is deep net so it is end to end shouldn't go through dimention reduction
    if cls_method == 'mlp':
        showFeat = False
        dimred_method = 'na'
    featuresNameScore = []
    # INITIALIZING
    cls_accuracy = cls_sensivity = cls_specificity = cls_ppv = cls_npv = cls_cm = 0
    cls_predicted = cls_expected = cls_aligned_subs = cls_confidence_scores = []
    [subjects_data, subjects_Minus, feature_names, X_data, y_data, X_MinusX_balanced, y_Minusy_balanced,
     UnlabeledData_original] = Data_load_normalize(age, dataflag, dimred_method, unlabel, fix_corrupt, corr,
                                                   pre_process, stat, mergeWith, gestAge, train_balanced)

    print '\n------------------X_MinusX_balanced Dimension --------------------', X_MinusX_balanced.shape

    kf = KFold(len(y_data), n_folds=no_folds, random_state=7)
    divideTest = KFold(len(y_Minusy_balanced), n_folds=no_folds,random_state=7)
    MinusX_dic = dict()
    for k, (firstPart_index, testMinus_index) in enumerate(divideTest):
        X_MinusX_balanced_test = X_MinusX_balanced[testMinus_index]
        y_Minusy_balanced_test = y_Minusy_balanced[testMinus_index]
        subjects_Minus_test = subjects_Minus[testMinus_index]

        MinusX_dic[k] = (subjects_Minus_test, X_MinusX_balanced_test, y_Minusy_balanced_test)

    for k, (train_index, test_index) in enumerate(kf):
        print '\n\n...........................................performing Fold #%d...............................................' % (
        k + 1)

        subjects_Minus_test, X_MinusX_balanced_test, y_Minusy_balanced_test = MinusX_dic[k]
        UnlabeledData = UnlabeledData_original
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        subjects_train, subjects_test = subjects_data[train_index], subjects_data[test_index]

        if test_balanced:
            pass
        else:
            X_test = np.vstack((X_test, X_MinusX_balanced_test))
            y_test = np.hstack((y_test, y_Minusy_balanced_test))
            print '\n------------------X_test Dimension --------------------', X_test.shape
            subjects_test = np.hstack((subjects_test, subjects_Minus_test))

        At_Risk_no_train = sum(y_train)
        Typical_no_train = len(y_train) - sum(y_train)
        print "Typical number Train=%d, At Risk number Train =%d" % (Typical_no_train, At_Risk_no_train)

        At_Risk_no_test = sum(y_test)
        Typical_no_test = len(y_test) - sum(y_test)
        print "Typical number Test=%d, At Risk number Test =%d" % (Typical_no_test, At_Risk_no_test)

        ratio = Typical_no_train / At_Risk_no_train - 1
        # ratio = float(At_Risk_no_train)/float(Typical_no_train)
        # ratio = 5.0
        if synthetic:
            X_train, y_train = OverSampling('SMOTE', False, X_train, y_train, ratio)
        # X_train,y_train = UnderSampling('cluster_cen',False,X_train,y_train)
        # X_train,y_train = OverUnder_Sampling('SMOTE_TOMEK',False,X_train,y_train,ratio)
        # X_train,y_train = EnsembleSampling('EasyEnsemble',False,X_train,y_train)

        if unlabel:

            X_train, X_test, UnlabeledData, featNameScore, threshold = DimensionReduction(feature_names, dimred_method,
                                                                                          X_test, X_train, y_train,
                                                                                          no_dims, UnlabeledData,
                                                                                          dataflag, showFeat)
        else:
            X_train, X_test, [], featNameScore, threshold = DimensionReduction(feature_names, dimred_method, X_test,
                                                                               X_train, y_train, no_dims, [], dataflag,
                                                                               showFeat)

        print "Dimensions after ", dimred_method, " is X_train Dimensions =", X_train.shape, 'X_test Dimensions=', X_test.shape
        # X_train, X_test = X_new[train_index], X_new[test_index]
        # print X_train.shape,X_test.shape
        if showFeat:
            featuresNameScore = featuresNameScore + featNameScore
        # print featuresNameScore
        ###the reason we put this is only the purpose of program running
        if stat == 1:
            fnames = ReturnFeatsScores(featNameScore, float("-inf"), showFeat, 'coef')

            print 'f_length=', len(fnames), 'fnames=', fnames
            statistical_analysis(X_train, y_train, fnames)
            return
        [method_name, cls_accuracy_, cls_sensivity_, cls_specificity_, cls_ppv_, cls_npv_, cls_cm_, predicted, expected,
         aligned_subs, confidence_scores] = select_classifier(binary, cls_method, X_train, y_train, X_test, y_test,
                                                              subjects_test, ver, UnlabeledData)

        cls_confidence_scores = cls_confidence_scores + list(confidence_scores)
        cls_predicted = cls_predicted + list(predicted)
        cls_expected = cls_expected + list(expected)
        cls_aligned_subs = cls_aligned_subs + list(aligned_subs)

        cls_accuracy += cls_accuracy_
        cls_sensivity += cls_sensivity_
        cls_specificity += cls_specificity_
        cls_ppv += cls_ppv_
        cls_npv += cls_npv_
        cls_cm += cls_cm_

    if showFeat:
        ShowFinalFeats(featuresNameScore, threshold, showFeat, 'coef')

    cls_df = print_kfold(method_name, cls_confidence_scores, cls_accuracy, cls_sensivity, cls_specificity, cls_ppv,
                         cls_npv, cls_cm, cls_predicted, cls_expected, cls_aligned_subs, no_folds)
    cls_df.to_csv('./results/%s%s_%s_%s_%s.csv' % (cls_method, mergeWith, age, dataflag, dimred_method), index=False)
    return cls_df


# Main script
#######################################################################################################################
# age = 'neo','1year','2year','long'
# cls_method=
#		'lr','nb','rbf','dt','knn',
#		'rfc','linear_svm','poly_svm'

# dimred_method = 'na'
# 		  'pls','lda'
#		  'nmf','lasso','ridge','elasticNet'
#		  'featSel_LASSO','featSel_ELASTIC','featSel_RIDGE','nnsc'
#		  'featSel_SVM','tree_based'
#		  'stabilitySel_lasso','stabilitySel_logistic'
#	          'pca','sparsepca','ica','randomForest',
#			'mlp'

# stat = 0,1 if 1 we do the statistical anlysis

# tunning_ver = 0,1,2 shows information during the tunning based on the verbos value

# print_folds =0,1 if 0 don't show classification info during each fold

# dataflag ='all','sub','fib','fibEnig','ZeroComp','LTComp','delta','deltaSub','deltaSub2y1y','deltaSub1yNeo' if 'all' the whole brain_cognitive data is loaded and if 'sub' the selected subset (caud/frontal/cigulum)
# if 'fib' the Tractography data will be loaded(fa,ad,md,rd) if 'fibEnig' The Fiber and Enigma data will be merged
# if 'ZeroComp' then the uncomplete data while the null elements changed to 0

# pre_process = 'norm','zscore' The default value is zscore

# unlabel = True , False , the default is True when it is false for unsuprevised dimension reduction we don't use unlabled data

# synthetic = True , False, the default is True when it is false we don't build any synthetic data

# corr = True, False, if corr=True then it visualise the correlation between features

# fix_corrupt= True, False if True will fix the number(digits) read from csv file cell by cell, to change the numbers that have two decimal points or more

# mergeWith = '' , 'age_dataflag'

# gestAge =True, False includes gestAge as one of our features for recognition


#	kappa_scorer = make_scorer(cohen_kappa_score)
# scoring=kappa_scorer
# scoring = 'precision_weighted'
# scoring = 'accuracy'

# binary = True , False, if it is True then the input data will be change to binary to reduce noise

system('clear')

cls_method = 'lr'
no_folds = 5
no_dims = 20

tunning_ver = 0
print_folds = 0
pre_process = 'zscore'
unlabel = True
synthetic = True
corr = False
fix_corrupt = False
showFeat = True
scoring = 'accuracy'
train_balanced = True
test_balanced = False
age_StructMRI = 'long'
#dimred_method_StructMRI = 'na'
#dimred_method_StructMRI = 'pca'
dimred_method_StructMRI = 'stabilitySel_lasso'

dataflag_StructMRI = 'deltaSub1yNeo'
mergeWith_StructMRI = '2year_sub'
gestAge_StructMRI = True
print_confidence = False
binary = False
#this is to find R2_scores and we don't use its "Pred_StructMRI_ " output

stat =0
if stat:
    no_folds = 2
    Pred_StructMRI_ = classification( binary, age_StructMRI, no_folds , no_dims, dimred_method_StructMRI,stat,
                                tunning_ver, print_folds, dataflag_StructMRI, pre_process, unlabel, synthetic, corr,
                                fix_corrupt, showFeat, scoring, mergeWith_StructMRI, gestAge_StructMRI, train_balanced,
                               test_balanced, print_confidence, cls_method)
stat = 0
no_folds = 5
gestAge_StructMRI = False
Pred_StructMRI_noGes = classification(binary, age_StructMRI, no_folds, no_dims, dimred_method_StructMRI, stat,
                                tunning_ver, print_folds, dataflag_StructMRI, pre_process, unlabel, synthetic, corr,
                                fix_corrupt, showFeat, scoring, mergeWith_StructMRI, gestAge_StructMRI, train_balanced,
                                test_balanced, print_confidence, cls_method)
gestAge_StructMRI = True
Pred_StructMRI = classification(binary, age_StructMRI, no_folds, no_dims, dimred_method_StructMRI, stat,
                                tunning_ver, print_folds, dataflag_StructMRI, pre_process, unlabel, synthetic, corr,
                                fix_corrupt, showFeat, scoring, mergeWith_StructMRI, gestAge_StructMRI, train_balanced,
                                test_balanced, print_confidence, cls_method)

age_DTI = '2year'
dimred_method_DTI = 'na'
dataflag_DTI = 'fibEnig'
mergeWith_DTI = ''
gestAge_DTI = False
showFeat = False
stat = 0
Pred_DTI = classification( binary, age_DTI, no_folds, no_dims, dimred_method_DTI, stat, tunning_ver,
                          print_folds, dataflag_DTI, pre_process, unlabel, synthetic, corr, fix_corrupt, showFeat,
                          scoring, mergeWith_DTI, gestAge_DTI, train_balanced, test_balanced, print_confidence,
                          cls_method)

featureFusion(Pred_StructMRI, Pred_DTI,Pred_StructMRI_noGes)

