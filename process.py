from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def preprocessNum(dataTrain):
    numeric_cols = [col for col in dataTrain.columns if dataTrain[col].dtype in ['float64', 'int64']]
    imputer = SimpleImputer(strategy = 'mean')
    dataTrain[numeric_cols] = imputer.fit_transform(dataTrain[numeric_cols])
    return(pd.DataFrame(dataTrain))

def preprocessCat(dataTrain): 
    categorical_cols = [col for col in dataTrain.columns if dataTrain[col].dtype == 'object']
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    #encoder = OneHotEncoder(handle_unknown='ignore')
    dataTrain[categorical_cols] = imputer.fit_transform(dataTrain[categorical_cols])
    dataTrain[categorical_cols] = encoder.fit_transform(dataTrain[categorical_cols])
    return(pd.DataFrame(dataTrain))
