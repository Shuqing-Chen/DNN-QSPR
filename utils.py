import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import json
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import utils
import warnings

from functions import *

def init_data(random_state_1=42,random_state_2=7):
    with open('data_all_180814.json', 'r',encoding="UTF-8") as f:
        data=json.load(f)
    x=[]
    y=[]
    t=[]
    # x0=[]
    # y0=[]
    for i in data['ss']:
        y.append(i[2]/100) #
        tem=[]
        for k in range(10):
            tem.append(di2(data['base'][i[0]][k],data['base'][i[1]][k]))
        x.append(tem)
        if y[-1]<0.01:
            t.append(0)
        elif y[-1]<0.1:
            t.append(1)
        elif y[-1]<0.2:
            t.append(2)
        elif y[-1]<0.4:
            t.append(3)
        elif y[-1]<0.8:
            t.append(4)
        else:
            t.append(5)
    attrs=data['desc']
    xy = pd.DataFrame(x,columns=attrs)
    xy['ss'] = y
    xy['t'] = t

    xy_train,xy_test = train_test_split(xy, test_size=0.2, shuffle=True, random_state=random_state_1, stratify=xy['t'])
    xy_valid,xy_test = train_test_split(xy_test, test_size=0.5, shuffle=True, random_state=random_state_2, stratify=xy_test['t'])
    # xy_train = xy.sample(frac=0.6)
    # xy_left = xy[~xy.index.isin(xy_train.index)]
    # xy_valid = xy_left.sample(frac=0.5)
    # xy_test = xy_left[~xy_left.index.isin(xy_valid.index)]

    xy_train = xy_train.reset_index(drop=True)
    xy_valid = xy_valid.reset_index(drop=True)
    xy_test = xy_test.reset_index(drop=True)
    print(len(xy),len(xy_train),len(xy_test),len(xy_valid))
    return xy,xy_train,xy_valid,xy_test

def init_data_2(random_state=42):
    with open('data_all_180814.json', 'r',encoding="UTF-8") as f:
        data=json.load(f)
    x=[]
    y=[]
    t=[]
    for i in data['ss']:
        y.append(i[2]/100) #
        tem=[]
        for k in range(10):
            tem.append(di2(data['base'][i[0]][k],data['base'][i[1]][k]))
        x.append(tem)
        if y[-1]<0.01:
            t.append(0)
        elif y[-1]<0.1:
            t.append(1)
        elif y[-1]<0.2:
            t.append(2)
        elif y[-1]<0.4:
            t.append(3)
        elif y[-1]<0.8:
            t.append(4)
        else:
            t.append(5)
    attrs=data['desc']
    xy = pd.DataFrame(x,columns=attrs)
    xy['ss'] = y
    xy['t'] = t
#     xy_train = xy.sample(frac=0.8)
#     xy_test = xy[~xy.index.isin(xy_train.index)]
    xy_train,xy_test = train_test_split(xy, test_size=0.2, shuffle=True, random_state=random_state, stratify=xy['t'])
    xy_train = xy_train.reset_index(drop=True)
    xy_test = xy_test.reset_index(drop=True)
    print(len(xy_train),len(xy_test))
    return xy,xy_train,xy_test

def transform(xy,attrs,model_arguments={}):
    y = xy['ss']
    if not model_arguments:
        for d in attrs:
            x_d = xy[d]
            x_m,y_m = data_select2(x_d,y)
            const = [1,1,1]
            bounds = ([0,0,0],[np.inf,np.inf,np.inf])
            res = least_squares(loss_coff, const,bounds=bounds,args=(x_m,y_m,model0) )
            x_line = np.arange(min(x_m),(max(x_m)),(max(x_m)-min(x_m))/300)
            y_line = model0(x_line,res.x)
            y_p = model0(x_m,res.x)
            model_arguments[d] = res.x
            # draw_plot([x_line,x_m],[y_line,y_m])
            print(d,res.x,loss_coff(res.x,x_m,y_m,model0),r2_score(y_p,y_m))
    attrs2 = [i+'2' for i in attrs]
    xy_transform = pd.DataFrame(np.array([model0(xy[d].tolist(),model_arguments[d]) for d in attrs]).T,columns=attrs2)
    xy_transform['ss'] = y.tolist()
    return xy_transform,model_arguments

class CusDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.dataLen = len(y)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.dataLen

class ReLU1(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU1, self).__init__(0., 1., inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
class Net(nn.Module):
    def __init__(self,din,dout,p1=0.3,p2=0.5):
        super(Net, self).__init__()
        h1, h2 = 64,128
        self.bn1 = nn.BatchNorm1d(din)
        self.fc1 = nn.Linear(din, h1) 
        self.do1 = nn.Dropout(p=p1, inplace=False)
        self.bn2 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.do2 = nn.Dropout(p=p2, inplace=False)
        self.fc3 = nn.Linear(h2, dout)
        self.relu1 = ReLU1()
        
        self.loss_fn_l1 = torch.nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.to(self.device)
    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.do1(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.do2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def fit(self,xy,attrs,target,iterations=3000,learning_rate = 0.001,batch_size=128):
        self.attrs = attrs
        self.target = target
        x = torch.tensor(xy[attrs].values).float()
        y = torch.tensor(xy[target].values).float()
        trainset = CusDataset(x,y)
        trainLoader = DataLoader(trainset, batch_size,shuffle=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for t in range(iterations):
            loss_total = 0
            for i, (inputs, labels) in enumerate(trainLoader):
                y_pred = self.forward(inputs)
                loss = self.loss_fn_l1(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
            if (t+1)%(iterations//10) == 0:
                print(t+1, loss_total/(i+1))

    def score(self,xy):
        x = torch.tensor(xy[self.attrs].values).float()
        y = torch.tensor(xy[self.target].values).float()
        y_pred = self.relu1(self.forward(x))
        error = self.loss_fn_l1(y_pred, y).item()
        return [error]

class MLs():
    def __init__(self):
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=166)
        self.predictors = [
            ensemble.RandomForestRegressor(n_estimators=30,max_depth=5),
            linear_model.Ridge(alpha=0.0001),
            linear_model.Lasso(alpha=0.0001,selection='random'),
            neighbors.KNeighborsRegressor(),
            svm.SVR(kernel='rbf',gamma='auto'),
            tree.DecisionTreeRegressor(max_depth = 5),
        ]
        self.alg_names = [i.__class__.__name__ for i in self.predictors]
    def train(self,xy,attrs,target):
        self.attrs = attrs
        self.target =target
        x = xy[attrs]
        y = np.ravel(xy[target])
        for predictor in self.predictors:
            predictor.fit(x,y)

    def score(self,xy):
        x = xy[self.attrs]
        y = np.ravel(xy[self.target])
        res_mae = []
        res_r2 = []
        for predictor in self.predictors:
            y_pred = predictor.predict(x)
            mae = mean_absolute_error(y,y_pred)
            r2 = predictor.score(x,y)
            res_mae.append(mae)
            res_r2.append(r2)
        return [res_mae,res_r2]

