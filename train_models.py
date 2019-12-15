from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn import utils
import warnings

def train_models(data,attrs,Target)->None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=166)
    #Machine Learning Algorithm (MLA) Selection and Initialization
    MLA = [
        #ensemble.AdaBoostRegressor(),
        #ensemble.BaggingRegressor(),
        #ensemble.ExtraTreesRegressor(n_estimators=10),
        #ensemble.GradientBoostingRegressor(),
        #XGBRegressor(),
        #gaussian_process.GaussianProcessRegressor(),
        ensemble.RandomForestRegressor(n_estimators=30,max_depth=5),
        linear_model.Ridge(alpha=0.0001),
        linear_model.Lasso(alpha=0.0001,selection='random'),
        neighbors.KNeighborsRegressor(),
        svm.SVR(kernel='rbf',gamma='auto'),
        tree.DecisionTreeRegressor(max_depth = 5),
        ]

    #split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    #note: this is an alternative to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = data[Target]
    data_target = utils.column_or_1d(MLA_predict.values.ravel(),warn=True)
    data_features = data[attrs]
    pd.options.mode.chained_assignment = None
    #index through MLA and save performance to table
    
    row_index = 0
    for alg in MLA:
        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    #     MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    #     print(MLA_name)
        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, data_features, data_target, cv  = cv_split,scoring = 'neg_mean_absolute_error', return_train_score=True)
        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
        

        #save MLA predictions - see section 6 for usage
        alg.fit(data_features, data_target)
        MLA_predict[MLA_name] = alg.predict(data_features)
        
        row_index+=1

    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    print(MLA_compare)
