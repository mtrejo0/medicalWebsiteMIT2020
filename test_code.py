import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from lr_model import get_model_output

# Tested with Python 3.6.8
# Tested with NumPy version 1.16.2
# Tested with sklearn version 0.21.1

# Load synthetic data
# File contains:
#     X_syn: Matrix of patient data
#         Rows are patients and columns are variables
#         Columns:
#         0: Transvalvular flow rate in mL/s
#         1: Mean pressure gradient across aortic valve in mmHg
#         2: Aortic valve area in cm^2
#         3: Congestive heart failure (CHF) at baseline (0 = No, 1 = Yes)
#         4: Myocardial infarction (MI) at baseline (0 = No, 1 = Yes)
#         5: Peripheral vascular disease (PVD) at baseline (0 = No, 1 = Yes)
#         6: Left ventricular segmental wall motion abnormality (0 = No, 1 = Yes)
#         7: Hyperlipidemia at baseline (0 = No, 1 = Yes)
#         8: Chronic kidney disease (CKD) at baseline (0 = No, 1 = Yes)
#         9: Posterior wall thickness (in millimeters)
#         10: Aortic sinus diameter (in millimeters)
#    Y_syn: Vector of labels (1 if mortality OR aortic valve replacement within 5 years, 0 otherwise)
f = open('./synthetic_data.pkl','rb')
X_syn,Y_syn = pickle.load(f)
f.close()
print(X_syn)
# Run synthetic data through the trained logistic regression model
yh_syn = get_model_output(X_syn)
print(yh_syn)
# Evaluate AUC on synthetic data
auc_syn = roc_auc_score(Y_syn,yh_syn)

# Printed AUC should be ~0.7563
print(auc_syn)
