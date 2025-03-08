import os
import joblib
import pandas as pd
import numpy as np
from src.data_cleaning.datacleaning import DataCleaning
from src.data_preprocessing.data_preprocessing import DataPreprocesing,DataEncoding, DataSpliting
from src.model_training.model import DNNModel, CNNModel, GRUModel, ModelCompair
from src.config import Config


def main():
    config = Config()
    
    print("...........Loading the Data........")
    
    df = pd.read_csv(config.DS_DATA_PATH)
    print("Data loaded successfully.")

    print("\nChecking first 5 rows of the dataset")
    print(df.head())

    print("\n.......Data preprocessing.........")
    dp = DataPreprocesing()
    df = dp.con_data(df)
    print("\Data preprocessing is done")

    print("\n...........Feature engineering..........")
    df = dp.feature_e(df)
    print("\nfeature engineering is done")
    
    print("\nChecking first 5 rows after the datapreprocessing and feature engineering")
    print(df.head())

    print("\nSaving the updated dataframe")
    save_dir = os.path.dirname(config.UPDATE_DATA_PATH)
    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(config.UPDATE_DATA_PATH, index=False)
    df.columns = df.columns.str.lower()
    df['disease'] = df['disease'].str.lower()
    print(df['disease'].unique().tolist())
    print("\nconverting column names to lowercases")
    df.columns = df.columns.str.lower()

    print("\nDataset shape:", df.shape)
    print("\nDataset description:")
    print(df.describe())
    
    print("\nDataset info:")
    df.info()
    
    print(".........Data Cleaning........")
    cl = DataCleaning()
    df = cl.drop_invalid_values(df)
    
    print("Data cleaning is done")

    print("........Encoding ........")
    df = DataEncoding.encodingdata(df)
    print("Encoding is applied")
    
    print("............Data Splitting.............")
    x_train, x_test, y_train, y_test = DataSpliting.dataspliting(df)  


    print("\n..............Model training...........")
    input_shape = 131 
    print("\n.........DNN model.........")
    best_dnn_model, y_pred_dnn, dnn_ac, dnn_ps, dnn_re, dnn_f1 = DNNModel.model_train(x_train, x_test, y_train, y_test, config)

    
    print("\n.......CNN model.......")
    best_cnn_model, y_pred_cnn, cnn_ac, cnn_ps, cnn_re, cnn_f1 = CNNModel.model_train(x_train, x_test, y_train, y_test, config, input_shape=input_shape)

    print("\n.........GRU model.........")
    best_gru_model, y_pred_gru, gru_ac, gru_ps, gru_re, gru_f1 = GRUModel.model_train(x_train, x_test, y_train, y_test, config, input_shape=input_shape)
   
    print("\n..........Comparing models.........")
    comp = ModelCompair(dnn_model=best_dnn_model,cnn_model=best_cnn_model,gru_model=best_gru_model,save_path=config.MODEL_SAVE_PATH)
    best_model_name, best_model_path = comp.compare_models(dnn_ac, cnn_ac, gru_ac)


if __name__ == "__main__":
    main()