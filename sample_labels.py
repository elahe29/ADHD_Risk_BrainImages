import pandas as pd
import numpy as np

labels = pd.read_csv('LABELS.csv')
at_risk_df = labels.loc[labels['Label']==1]
typical_df = labels.loc[labels['Label']==0]  

num_atRisk=len(at_risk_df)
num_typical=len(typical_df)

sample_lables_df = at_risk_df
print 'At Risk number of subjects:',len(sample_lables_df)

chooseFromTypical = 2*(float(num_atRisk)/num_typical)
random_typical_df = typical_df.sample(frac=chooseFromTypical)
print 'Typical number of subjects:',len(random_typical_df)

sample_lables_df = sample_lables_df.append(random_typical_df,ignore_index=True)
print 'Final number of subjects:',len(sample_lables_df)

sample_lables_df.to_csv('LABELS_balanced.csv',index=False)
