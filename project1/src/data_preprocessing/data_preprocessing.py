import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter


class DataPreprocesing:
    """
      Convert the unique symptoms into columns
    """

    def __init__(self):
        pass

    
    def con_data(self,df):
        unique_symptoms = set()
        for col in df.columns[1:]:
            unique_symptoms.update(df[col].dropna().unique())

        for symptom in unique_symptoms:
            df[symptom] = df.iloc[:, 1:].apply(lambda row: 1 if symptom in row.values else 0, axis=1)


        df = df.drop(columns=df.columns[1:18])

        return df
    
    def feature_e(self,df):
        for i, row in df.iterrows():
            disease = df.iloc[i, 0] 
            symptoms_for_disease = df.iloc[i, 1:18].dropna()
    
            for symptom in symptoms_for_disease:
                if symptom in df.columns:
                    df.loc[i, symptom] = 1
        
        
        return df



class DataEncoding:

    def __init__(self):
        pass

    def encodingdata(df):
        
        """
        Encoding for the categorical values

        """

        le = LabelEncoder()
        df['disease'] = le.fit_transform(df['disease'])


        return df

class DataSpliting:

    """
    After data preprocessing spliting data into x_train,y_train,x_test,y_test
    """
    @staticmethod
    def dataspliting(df): 
        x = df.drop('disease', axis=1)
        y = df['disease']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        print("x_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        return x_train, x_test, y_train, y_test




    