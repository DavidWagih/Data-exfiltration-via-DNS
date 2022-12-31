# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


#df = pd.read_csv("../data/training_dataset.csv")

def columns_to_drop_ret():
    return ["Label","sld","timestamp","longest_word"]

def prepare_data(df):
    columns_to_drop=columns_to_drop_ret()
    # Preparing data
    y=df["Label"]
    X=df.drop(columns_to_drop,axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y)
    
    return X_train,X_test,y_train,y_test

