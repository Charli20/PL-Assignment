import os
import joblib
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data_cleaning.datacleaning import RenameImg
from src.EDA.eda import EDA
from src.data_preprocessing.data_preprocessing import CropImg,DataSplitting
from src.model_training.model import CNN,VGG19Model,ModelCompare


def main():

    print('..........loading images..............')
    train_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\Training"
    test_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\Testing"   
    categories = {"glioma": "gl","meningioma": "men", "notumor": "notu","pituitary": "pitu"}

    print('.......Rename images.........')
    for category, prefix in categories.items():
        RenameImg.rename_images(os.path.join(train_dir, category), prefix)
    print('.........Rename testing images...........')
    for category, prefix in categories.items():
        RenameImg.rename_images(os.path.join(test_dir, category), prefix)
    print('.......finish renaming images....... ')
    print("........EDA........")
    EDA.plot_image_counts()
    print(".......Eda is finish......")
    print(".......Data Preprocessing........")
    crop = CropImg()
       
    cropped_train_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\cropped_data\Train"
    cropped_test_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\cropped_data\Test"

    categories = ["glioma", "meningioma", "notumor", "pituitary"]

    print("\n Cropping train images...")
    crop.process_images(train_dir, cropped_train_dir, categories) 
    print("\n Cropping test images...")
    crop.process_images(test_dir, cropped_test_dir, categories) 
    print("......Data preprocessing is finished.......")
    print(".......Data splitting........")
    val_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\cropped_data\Val"
    DataSplitting.split_testing_images(cropped_test_dir, val_dir, categories, split_size=300)
    print("......Data splitting is done.......")

    print("..........model trianing.........")
    print("...........custom cnnn model.............")
    cnn = CNN()
    test_new_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\cropped_data\Test"
    train_datagen_cnn = ImageDataGenerator(rescale=1./255)
    cnn_traindt = train_datagen_cnn.flow_from_directory(cropped_train_dir, target_size=(150,150), batch_size=32, class_mode='categorical')
    cnn_valdt = train_datagen_cnn.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode='categorical')
    cnn_testdt = train_datagen_cnn.flow_from_directory(test_new_dir,target_size=(150,150), batch_size=32,class_mode='categorical', shuffle=False)
    
    history_1 = cnn.train_and_evaluate(cnn_traindt, cnn_testdt)
    cnn.cnn_plot(history_1)
    cnn_ac, cnn_pr, cnn_re, cnn_f1 = cnn.predict_evaluate(cnn_testdt,cnn_valdt) 

    print('................Vgg19 model.............')
    vg19 = VGG19Model()
    train_datagen_vgg19 = ImageDataGenerator(rescale=1./255)
    vgg19_traindt = train_datagen_vgg19.flow_from_directory(cropped_train_dir, target_size=(128,128), batch_size=32, class_mode='categorical')
    vgg19_valdt = train_datagen_vgg19.flow_from_directory(val_dir, target_size=(128,128), batch_size=32, class_mode='categorical')
    vgg19_testdt = train_datagen_vgg19.flow_from_directory(test_new_dir,target_size=(128,128), batch_size=32,class_mode='categorical', shuffle=False)
 
    history_2 = vg19.train_and_evaluate(vgg19_traindt, vgg19_valdt) 
    vg19.vgg_plot(history_2)
    vg_ac, vg_pr, vg_re, vg_f1 = vg19.predict_evaluate(vgg19_testdt,vgg19_valdt)


    print(".............Selecting best model....................")
    model_compare = ModelCompare({"custom_cnn": cnn, "vgg19": vg19}, cnn_valdt) 
    model_compare.compare_models(vg_ac, vg_pr, vg_re, vg_f1, cnn_ac, cnn_pr, cnn_re, cnn_f1)
    print("\nBest model saved as bestmodel.keras")


if __name__ == "__main__":
    main()