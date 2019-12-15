import seaborn as sns
import itertools
import random
from scipy.stats import pearsonr
from train_models import train_models 
from utils import *
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold


# attrs = ["Fermi", "Bond", "Mass", "No", "Radius", "Boiling", "Melting", "Mendeleev", "Volume", "X"]
attrs = ["Fermi", "Radius", "Mendeleev", "X"]
attrs2 = ["Fermi2", "Radius2", "Mendeleev2", "X2"]
target = ["ss"]

xy_all,xy_train,xy_valid,xy_test = init_data()

xy_train_transform,model_arguments = transform(xy_train,attrs)

xy_valid_transform,_ = transform(xy_valid,attrs,model_arguments)

xy_test_transform,_ = transform(xy_test,attrs,model_arguments)

xy_train_all = xy_train_transform.join(xy_train[attrs])

xy_valid_all = xy_valid_transform.join(xy_valid[attrs])

xy_test_all = xy_test_transform.join(xy_test[attrs])

# print("\nTraining with DNN and raw data")
# dnn = Net(4,1)
# dnn.fit(xy_train,attrs,['ss'])
# dnn.score(xy_train)
# dnn.score(xy_valid)
# dnn.score(xy_test)

# print("\nTraining with DNN and transformed data")
# dnn2 = Net(4,1)
# dnn2.fit(xy_train_transform,attrs2,['ss'])
# dnn2.score(xy_train_transform)
# dnn2.score(xy_valid_transform)
# dnn2.score(xy_test_transform)

# print("\nTraining with DNN and transformed&raw data")
# dnn2 = Net(8,1)
# dnn2.fit(xy_train_all,attrs+attrs2,['ss'])
# dnn2.score(xy_train_all)
# dnn2.score(xy_valid_all)
# dnn2.score(xy_test_all)

print("\nTraining with general MLs and raw data")
mls = MLs()
mls.train(xy_train,attrs,['ss'])
res_train = mls.score(xy_train)
res_valid = mls.score(xy_valid)
res_test = mls.score(xy_test)
info = pd.DataFrame()
info["alg_names"] = mls.alg_names
info["mae_train"] = res_train
info["mae_valid"] = res_valid
info["mae_test"] = res_test
print(info)

print("\nTraining with general MLs and transformed data")
mls2 = MLs()
mls2.train(xy_train_transform,attrs2,['ss'])
res_train = mls2.score(xy_train_transform)
res_valid = mls2.score(xy_valid_transform)
res_test = mls2.score(xy_test_transform)
info = pd.DataFrame()
info["alg_names"] = mls.alg_names
info["mae_train"] = res_train
info["mae_valid"] = res_valid
info["mae_test"] = res_test
print(info)

print("\nTraining with general MLs and raw&transformed data")
mls2 = MLs()
mls2.train(xy_train_all,attrs2,['ss'])
res_train = mls2.score(xy_train_all)
res_valid = mls2.score(xy_valid_all)
res_test = mls2.score(xy_test_all)
info = pd.DataFrame()
info["alg_names"] = mls.alg_names
info["mae_train"] = res_train
info["mae_valid"] = res_valid
info["mae_test"] = res_test

print(info)