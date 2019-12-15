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
import time

def run():
    # attrs = ["Fermi", "Bond", "Mass", "No", "Radius", "Boiling", "Melting", "Mendeleev", "Volume", "X"]
    attrs = ["Fermi", "Radius", "Mendeleev", "X"]
    attrs2 = ["Fermi2", "Radius2", "Mendeleev2", "X2"]
    target = ["ss"]
    xy_all,xy_train,xy_test = init_data_2()

    ## init learners
    MLA = [
        ensemble.RandomForestRegressor(n_estimators=30,max_depth=5),
        linear_model.Ridge(alpha=0.0001),
        linear_model.Lasso(alpha=0.0001,selection='random'),
        neighbors.KNeighborsRegressor(),
        svm.SVR(kernel='rbf',gamma='auto'),
        tree.DecisionTreeRegressor(max_depth = 5),
        ]

    ## KFolder cross-validated
    KF=StratifiedKFold(n_splits=4,shuffle=True)
    
    mae_train = [0 for _ in range(7)]
    mae_valid = [0 for _ in range(7)]
    mae_test = [0 for _ in range(7)]
    
    mae_train_t = [0 for _ in range(7)]
    mae_valid_t = [0 for _ in range(7)]
    mae_test_t = [0 for _ in range(7)]
    
    mae_train_all = [0 for _ in range(7)]
    mae_valid_all = [0 for _ in range(7)]
    mae_test_all = [0 for _ in range(7)]
    
    data = xy_train
    for train_index,valid_index in KF.split(data,data['t']):
        model_train = data.iloc[train_index].reset_index(drop=True)
        model_valid = data.iloc[valid_index].reset_index(drop=True)
        x_train = model_train[attrs]
        y_train = model_train[target]
        x_valid = model_valid[attrs]
        y_valid = model_valid[target]

        model_train_t,transform_para = transform(model_train,attrs,{})
        model_valid_t,_ = transform(model_valid,attrs,transform_para)
        x_train_t = model_train_t[attrs2]
        y_train_t = model_train_t[target]
        x_valid_t = model_valid_t[attrs2]
        y_valid_t = model_valid_t[target]
        
        model_train_all = model_train_t.join(model_train[attrs])
        model_valid_all = model_valid_t.join(model_valid[attrs])
        x_train_all = model_train_all[attrs2+attrs]
        y_train_all = model_train_all[target]
        x_valid_all = model_valid_all[attrs2+attrs]
        y_valid_all = model_valid_all[target]

        dnn = Net(4,1,p1=0.5,p2=0.5)
        dnn.fit(model_train,attrs,target,iterations=3000,learning_rate = 0.001,batch_size=64)
        res = dnn.score(model_train)
        mae_train[0] += res[0]/KF.get_n_splits()
        res =dnn.score(model_valid)
        mae_valid[0] += res[0]/KF.get_n_splits()

        y_train = np.ravel(y_train)
        y_valid = np.ravel(y_valid)
        for index,ml in enumerate(MLA):
            ml.fit(x_train,y_train)
            y_pred = ml.predict(x_train)
            mae_train[index+1] += mean_absolute_error(y_train,y_pred)/KF.get_n_splits()
            y_pred = ml.predict(x_valid)
            mae_valid[index+1] += mean_absolute_error(y_valid,y_pred)/KF.get_n_splits()

        dnn = Net(4,1,p1=0.5,p2=0.7)
        dnn.fit(model_train_t,attrs2,target,iterations=3000,learning_rate = 0.001,batch_size=64)
        res = dnn.score(model_train_t)
        mae_train_t[0] += res[0]/KF.get_n_splits()
        res =dnn.score(model_valid_t)
        mae_valid_t[0] += res[0]/KF.get_n_splits()

        y_train_t = np.ravel(y_train_t)
        y_valid_t = np.ravel(y_valid_t)
        for index,ml in enumerate(MLA):
            ml.fit(x_train_t,y_train_t)
            y_pred = ml.predict(x_train_t)
            mae_train_t[index+1] += mean_absolute_error(y_train_t,y_pred)/KF.get_n_splits()
            y_pred = ml.predict(x_valid_t)
            mae_valid_t[index+1] += mean_absolute_error(y_valid_t,y_pred)/KF.get_n_splits()
        
        dnn = Net(8,1,p1=0.5,p2=0.7)
        dnn.fit(model_train_all,attrs2+attrs,target,iterations=3000,learning_rate = 0.001,batch_size=64)
        res = dnn.score(model_train_all)
        mae_train_all[0] += res[0]/KF.get_n_splits()
        res =dnn.score(model_valid_all)
        mae_valid_all[0] += res[0]/KF.get_n_splits()

        y_train_all = np.ravel(y_train_all)
        y_valid_all = np.ravel(y_valid_all)
        for index,ml in enumerate(MLA):
            ml.fit(x_train_all,y_train_all)
            y_pred = ml.predict(x_train_all)
            mae_train_all[index+1] += mean_absolute_error(y_train_all,y_pred)/KF.get_n_splits()
            y_pred = ml.predict(x_valid_all)
            mae_valid_all[index+1] += mean_absolute_error(y_valid_all,y_pred)/KF.get_n_splits()
    
    ##test
    model_train = xy_train
    model_test = xy_test
    x_train = model_train[attrs]
    y_train = model_train[target]
    x_test = model_test[attrs]
    y_test = model_test[target]

    model_train_t,transform_para = transform(model_train,attrs,{})
    model_test_t,_ = transform(model_test,attrs,transform_para)
    x_train_t = model_train_t[attrs2]
    y_train_t = model_train_t[target]
    x_test_t = model_test_t[attrs2]
    y_test_t = model_test_t[target]

    model_train_all = model_train_t.join(model_train[attrs])
    model_test_all = model_test_t.join(model_test[attrs])
    x_train_all = model_train_all[attrs2+attrs]
    y_train_all = model_train_all[target]
    x_test_all = model_test_all[attrs2+attrs]
    y_test_all = model_test_all[target]

    dnn = Net(4,1,p1=0.5,p2=0.5)
    dnn.fit(model_train,attrs,target,iterations=3000,learning_rate = 0.001,batch_size=64)
    res =dnn.score(model_test)
    mae_test[0] = res[0]

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    for index,ml in enumerate(MLA):
        ml.fit(x_train,y_train)
        y_pred = ml.predict(x_test)
        mae_test[index+1] = mean_absolute_error(y_test,y_pred)

    dnn = Net(4,1,p1=0.5,p2=0.7)
    dnn.fit(model_train_t,attrs2,target,iterations=3000,learning_rate = 0.001,batch_size=64)
    res =dnn.score(model_test_t)
    mae_test_t[0] = res[0]

    y_train_t = np.ravel(y_train_t)
    y_test_t = np.ravel(y_test_t)
    for index,ml in enumerate(MLA):
        ml.fit(x_train_t,y_train_t)
        y_pred = ml.predict(x_test_t)
        mae_test_t[index+1] = mean_absolute_error(y_test_t,y_pred)
    
    dnn = Net(8,1,p1=0.5,p2=0.7)
    dnn.fit(model_train_all,attrs2+attrs,target,iterations=3000,learning_rate = 0.001,batch_size=64)
    res =dnn.score(model_test_all)
    mae_test_all[0] = res[0]

    y_train_all = np.ravel(y_train_all)
    y_test_all = np.ravel(y_test_all)
    for index,ml in enumerate(MLA):
        ml.fit(x_train_all,y_train_all)
        y_pred = ml.predict(x_test_all)
        mae_test_all[index+1] = mean_absolute_error(y_test_all,y_pred)
            
    info = pd.DataFrame()
    info["alg_names"] = ["dnn","RF","Ridge","Lasso","KNN","SVR","DT"]
    
    info["mae_train"] = mae_train
    info["mae_valid"] = mae_valid
    info["mae_test"] = mae_test
    
    info["mae_train_t"] = mae_train_t
    info["mae_valid_t"] = mae_valid_t
    info["mae_test_t"] = mae_test_t
    
    info["mae_train_all"] = mae_train_all
    info["mae_valid_all"] = mae_valid_all
    info["mae_test_all"] = mae_test_all
    
    print(info)
    return info

if __name__ == "__main__":
    time_start=time.time()
    run()
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    