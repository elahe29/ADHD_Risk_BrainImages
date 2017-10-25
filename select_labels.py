import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BRIEF = pd.read_csv('BRIEF_ADHD.csv')
BASC = pd.read_csv('BASC_ADHD.csv')

BRIEF = BRIEF[['subjectID','BRIEF6yr_BRI_T','BRIEF6yr_WM_T','BRIEF6yr_MI_T']]
BASC = BASC[['subjectID','PR_BASC6yr_HYP_T','PR_BASC6yr_ExProbCOMP_T','PR_BASC6yr_BSI_COMP_T','PR_BASC6yr_AttProb_T']]

BRIEFx,BRIEFy=BRIEF.shape
OutRange_BRIEF=np.zeros((BRIEFx,BRIEFy-1),dtype='int64')
NumOneSTDAway_BRIEF=[]

BASCx,BASCy=BASC.shape
OutRange_BASC=np.zeros((BASCx,BASCy-1),dtype='int64')
NumOneSTDAway_BASC=[]


"""mean_BRIEF = BRIEF.mean(axis=0)
mean_BASC = BASC.mean(axis=0)
print mean_BRIEF
print mean_BASC

std_BRIEF = BRIEF.std(axis=0)
std_BASC = BASC.std(axis=0)
print std_BRIEF
print std_BASC"""

mean_BRIEF = {'BRIEF6yr_BRI_T':50,'BRIEF6yr_WM_T':50,'BRIEF6yr_MI_T':50}
std_BRIEF = {'BRIEF6yr_BRI_T':10,'BRIEF6yr_WM_T':10,'BRIEF6yr_MI_T':10}
BRIEF_STD_MUL = 3.0/3

mean_BASC = {'PR_BASC6yr_HYP_T':50,'PR_BASC6yr_ExProbCOMP_T':50,'PR_BASC6yr_BSI_COMP_T':50,'PR_BASC6yr_AttProb_T':50}
std_BASC = {'PR_BASC6yr_HYP_T':10,'PR_BASC6yr_ExProbCOMP_T':10,'PR_BASC6yr_BSI_COMP_T':10,'PR_BASC6yr_AttProb_T':10}
BASC_STD_MUL = 1.0

figbrief=plt.figure(1)
figbrief.canvas.set_window_title('BRIEF') 

plt.subplot(311)

plt.plot(np.zeros(BRIEFx),BRIEF['BRIEF6yr_BRI_T'],'ro')

plt.plot([-1,1],[mean_BRIEF['BRIEF6yr_BRI_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_BRI_T'],mean_BRIEF['BRIEF6yr_BRI_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_BRI_T']],'r-'); 
plt.plot([-1,1],[mean_BRIEF['BRIEF6yr_BRI_T'],mean_BRIEF['BRIEF6yr_BRI_T']],'k-');

OutRange_BRIEF[:,0]=(BRIEF['BRIEF6yr_BRI_T']>(mean_BRIEF['BRIEF6yr_BRI_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_BRI_T']))
NumOneSTDAway_BRIEF.append(np.sum(OutRange_BRIEF[:,0]))

plt.title('BRI- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BRIEF[0],NumOneSTDAway_BRIEF[0]*100.0/BRIEFx));


plt.subplot(312)

plt.plot(np.zeros(BRIEFx),BRIEF['BRIEF6yr_WM_T'],'bo')

plt.plot([-1,1],[mean_BRIEF['BRIEF6yr_WM_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_WM_T'],mean_BRIEF['BRIEF6yr_WM_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_WM_T']],'r-'); 
plt.plot([-1,1],[mean_BRIEF['BRIEF6yr_WM_T'],mean_BRIEF['BRIEF6yr_WM_T']],'k-');

OutRange_BRIEF[:,1]=(BRIEF['BRIEF6yr_WM_T']>(mean_BRIEF['BRIEF6yr_WM_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_WM_T']))
NumOneSTDAway_BRIEF.append(np.sum(OutRange_BRIEF[:,1]))

plt.title('Working Memory- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BRIEF[1],NumOneSTDAway_BRIEF[1]*100.0/BRIEFx));

plt.subplot(313)

plt.plot(np.zeros(BRIEFx),BRIEF['BRIEF6yr_MI_T'],'go')

plt.plot([-1,1],[mean_BRIEF['BRIEF6yr_MI_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_MI_T'],mean_BRIEF['BRIEF6yr_MI_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_MI_T']],'r-'); 
plt.plot([-1,1],[mean_BRIEF['BRIEF6yr_MI_T'],mean_BRIEF['BRIEF6yr_MI_T']],'k-');

OutRange_BRIEF[:,2]=(BRIEF['BRIEF6yr_MI_T']>(mean_BRIEF['BRIEF6yr_MI_T']+(BRIEF_STD_MUL)*std_BRIEF['BRIEF6yr_MI_T']))
NumOneSTDAway_BRIEF.append(np.sum(OutRange_BRIEF[:,2]))

plt.title('MetaCognitive Index- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BRIEF[2],NumOneSTDAway_BRIEF[2]*100.0/BRIEFx));

OutRange_MetaWM_BRIEF_ = np.logical_and(OutRange_BRIEF[:,1],OutRange_BRIEF[:,2])
OutRange_MetaWM_BRIEF_no = np.sum(OutRange_MetaWM_BRIEF_)
OutRange_MetaWM_BRIEF_pcn = np.sum(OutRange_MetaWM_BRIEF_)*100.0/BRIEFx
print "OutRange_MetaWM_BRIEF: numOfCases=%d, Percentile=%f" %(OutRange_MetaWM_BRIEF_no,OutRange_MetaWM_BRIEF_pcn)

OutRange_MetaorWM_BRIEF_ = np.logical_or(OutRange_BRIEF[:,1],OutRange_BRIEF[:,2])
OutRange_MetaorWM_BRIEF_no = np.sum(OutRange_MetaorWM_BRIEF_)
OutRange_MetaorWM_BRIEF_pcn = np.sum(OutRange_MetaorWM_BRIEF_)*100.0/BRIEFx
print "OutRange_MetaorWM_BRIEF: numOfCases=%d, Percentile=%f" %(OutRange_MetaorWM_BRIEF_no,OutRange_MetaorWM_BRIEF_pcn)

OutRange_all_BRIEF_ = np.logical_and(np.logical_and(OutRange_BRIEF[:,0],OutRange_BRIEF[:,1]),OutRange_BRIEF[:,2])
OutRange_all_BRIEF_no = np.sum(OutRange_all_BRIEF_)
OutRange_all_BRIEF_pcn = np.sum(OutRange_all_BRIEF_)*100.0/BRIEFx
print "OutRange_all_BRIEF: numOfCases=%d, Percentile=%f" %(OutRange_all_BRIEF_no,OutRange_all_BRIEF_pcn)

OutRange_any_BRIEF_ = np.logical_or(np.logical_or(OutRange_BRIEF[:,0],OutRange_BRIEF[:,1]),OutRange_BRIEF[:,2])
OutRange_any_BRIEF_no = np.sum(OutRange_any_BRIEF_)
OutRange_any_BRIEF_pcn = np.sum(OutRange_any_BRIEF_)*100.0/BRIEFx
print "OutRange_any_BRIEF: numOfCases=%d, Percentile=%f" %(OutRange_any_BRIEF_no,OutRange_any_BRIEF_pcn)

plt.show(block=False)
##################################################################BASC###############################################################
figbasc=plt.figure(2)
figbasc.canvas.set_window_title('BASC') 

plt.subplot(411)

plt.plot(np.zeros(BASCx),BASC['PR_BASC6yr_HYP_T'],'ro')

plt.plot([-1,1],[mean_BASC['PR_BASC6yr_HYP_T']+ (BASC_STD_MUL)*std_BASC['PR_BASC6yr_HYP_T'],mean_BASC['PR_BASC6yr_HYP_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_HYP_T']],'r-'); 
plt.plot([-1,1],[mean_BASC['PR_BASC6yr_HYP_T'],mean_BASC['PR_BASC6yr_HYP_T']],'k-');

OutRange_BASC[:,0]=(BASC['PR_BASC6yr_HYP_T']>(mean_BASC['PR_BASC6yr_HYP_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_HYP_T']))
NumOneSTDAway_BASC.append(np.sum(OutRange_BASC[:,0]))

plt.title('Hyperactivity- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BASC[0],NumOneSTDAway_BASC[0]*100.0/BASCx))


plt.subplot(412)

plt.plot(np.zeros(BASCx),BASC['PR_BASC6yr_ExProbCOMP_T'],'bo')

plt.plot([-1,1],[mean_BASC['PR_BASC6yr_ExProbCOMP_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_ExProbCOMP_T'],mean_BASC['PR_BASC6yr_ExProbCOMP_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_ExProbCOMP_T']],'r-'); 
plt.plot([-1,1],[mean_BASC['PR_BASC6yr_ExProbCOMP_T'],mean_BASC['PR_BASC6yr_ExProbCOMP_T']],'k-');

OutRange_BASC[:,1]=(BASC['PR_BASC6yr_ExProbCOMP_T']>(mean_BASC['PR_BASC6yr_ExProbCOMP_T']+ (BASC_STD_MUL)*std_BASC['PR_BASC6yr_ExProbCOMP_T']))
NumOneSTDAway_BASC.append(np.sum(OutRange_BASC[:,1]))

plt.title('Externalizing Problem- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BASC[1],NumOneSTDAway_BASC[1]*100.0/BASCx))
plt.subplot(413)

plt.plot(np.zeros(BASCx),BASC['PR_BASC6yr_BSI_COMP_T'],'go')

plt.plot([-1,1],[mean_BASC['PR_BASC6yr_BSI_COMP_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_BSI_COMP_T'],mean_BASC['PR_BASC6yr_BSI_COMP_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_BSI_COMP_T']],'r-'); 
plt.plot([-1,1],[mean_BASC['PR_BASC6yr_BSI_COMP_T'],mean_BASC['PR_BASC6yr_BSI_COMP_T']],'k-');

OutRange_BASC[:,2]=(BASC['PR_BASC6yr_BSI_COMP_T']>(mean_BASC['PR_BASC6yr_BSI_COMP_T']+ (BASC_STD_MUL)*std_BASC['PR_BASC6yr_BSI_COMP_T']))
NumOneSTDAway_BASC.append(np.sum(OutRange_BASC[:,2]))

plt.title('BSI- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BASC[2],NumOneSTDAway_BASC[2]*100.0/BASCx))


plt.subplot(414)
plt.plot(np.zeros(BASCx),BASC['PR_BASC6yr_AttProb_T'],'co')

plt.plot([-1,1],[mean_BASC['PR_BASC6yr_AttProb_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_AttProb_T'],mean_BASC['PR_BASC6yr_AttProb_T']+(BASC_STD_MUL)*std_BASC['PR_BASC6yr_AttProb_T']],'r-'); 
plt.plot([-1,1],[mean_BASC['PR_BASC6yr_AttProb_T'],mean_BASC['PR_BASC6yr_AttProb_T']],'k-');

OutRange_BASC[:,3]=(BASC['PR_BASC6yr_AttProb_T']>(mean_BASC['PR_BASC6yr_AttProb_T']+ (BASC_STD_MUL)*std_BASC['PR_BASC6yr_AttProb_T']))
NumOneSTDAway_BASC.append(np.sum(OutRange_BASC[:,3]))

plt.title('Attention Problem- Number of cases at Risk=%d and Percentile=%f' %(NumOneSTDAway_BASC[3],NumOneSTDAway_BASC[3]*100.0/BASCx))
plt.show(block=False)

OutRange_all_BASC_ =   np.logical_and(np.logical_and(OutRange_BASC[:,0],OutRange_BASC[:,1]),np.logical_and(OutRange_BASC[:,2],OutRange_BASC[:,3]))
OutRange_all_BASC_no = sum(OutRange_all_BASC_)
OutRange_all_BASC_pcn = sum(OutRange_all_BASC_)*100.0/BASCx
print "OutRange_all_BASC: numOfCases=%d, Percentile=%f" %(OutRange_all_BASC_no,OutRange_all_BASC_pcn)

OutRange_any_BASC_ = np.logical_or(np.logical_or(OutRange_BASC[:,0],OutRange_BASC[:,1]),np.logical_or(OutRange_BASC[:,2],OutRange_BASC[:,3]))
OutRange_any_BASC_no = sum(OutRange_any_BASC_)
OutRange_any_BASC_pcn = sum(OutRange_any_BASC_)*100.0/BASCx
print "OutRange_any_BASC: numOfCases=%d, Percentile=%f" %(OutRange_any_BASC_no,OutRange_any_BASC_pcn)


OutRange_all_ = np.logical_and(OutRange_all_BASC_,OutRange_all_BRIEF_)
OutRange_all_no = sum(OutRange_all_)
OutRange_all_pcn = sum(OutRange_all_)*100.0/BASCx
print "OutRange_all: numOfCases=%d, Percentile=%f" %(OutRange_all_no,OutRange_all_pcn)
 
OutRange_any_ = np.logical_or(OutRange_any_BASC_,OutRange_any_BRIEF_)
OutRange_any_no = sum(OutRange_any_)
OutRange_any_pcn = sum(OutRange_any_)*100.0/BASCx
print "OutRange_any: numOfCases=%d, Percentile=%f" %(OutRange_any_no,OutRange_any_pcn)
 
OutRange_all_any_ = np.logical_or(OutRange_all_BASC_,OutRange_all_BRIEF_)
OutRange_allBasc_OR_allBRIEF_no = sum(OutRange_all_any_)
OutRange_allBasc_OR_allBRIEF_pcn = sum(OutRange_all_any_)*100.0/BASCx
print "OutRange_allBasc_OR_allBRIEF: numOfCases=%d, Percentile=%f" %(OutRange_allBasc_OR_allBRIEF_no,OutRange_allBasc_OR_allBRIEF_pcn)
 
OutRange_any_all_= np.logical_and(OutRange_any_BASC_,OutRange_any_BRIEF_)
OutRange_anyBASC_AND_anyBRIEF_no = sum(OutRange_any_all_)
OutRange_anyBASC_AND_anyBRIEF_pcn = sum(OutRange_any_all_)*100.0/BASCx
print "OutRange_anyBASC_AND_anyBRIEF: numOfCases=%d, Percentile=%f" %(OutRange_anyBASC_AND_anyBRIEF_no,OutRange_anyBASC_AND_anyBRIEF_pcn)  

# Change The True, False labels to 0 and 1
# Label based on BRIEF : Metacognitive and Working Memory
OutRange_MetaWM_BRIEF_ = [int(element) for element in OutRange_MetaWM_BRIEF_]
LABELS_dic= {'subjectID':BRIEF['subjectID'],'Label':OutRange_MetaWM_BRIEF_}
LABELS = pd.DataFrame(LABELS_dic)
LABELS.to_csv('LABELS.csv',index=False)
plt.show()
