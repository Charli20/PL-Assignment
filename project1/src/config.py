import os

class Config:
    DS_DATA_PATH = r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\raw_dataset\dataset.csv'
    DS1_PATH = r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\raw_dataset\symptom_Description.csv'
    DS2_PATH = r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\raw_dataset\symptom_precaution.csv'
    UPDATE_DATA_PATH = r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\created\update_df.csv'
    NEW_DATA_PATH = r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\created\new_df.csv'
    MODEL_SAVE_PATH = r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\model'
    

    DNN_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "dnn_model.pkl")  
    CNN_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "cnn_model.pkl")

