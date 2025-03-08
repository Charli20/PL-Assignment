import pandas as pd 
import numpy as np 


class DataCleaning:

    def __init__(self):
        pass
    
    def drop_invalid_values(self,df):
        """
        Removing null values in the dataset.

        """
        
        print("\nCheking null values in the dataset",df.isna().sum())
        
        df.dropna(inplace=True)

        print("\nDroping duplicate values in the dataset")
        df.drop_duplicates(inplace=True)

        return df

    