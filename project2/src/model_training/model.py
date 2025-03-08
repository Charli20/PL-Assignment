import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class CNN:
    def __init__(self):
        self.model = Sequential([
            Input(shape=(150, 150, 3)),
            Conv2D(32, (3,3), activation='relu'),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.3),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(64, (3,3), activation='relu'),
            Dropout(0.3),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),  
            MaxPooling2D(2,2),
            Dropout(0.3),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='softmax')
        ])
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    def train_and_evaluate(self, train_data, val_data):
        model_save_path = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\model\CNNmodel.keras"
        ch = ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
        es = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
        re_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.0001)
        
        history = self.model.fit(train_data, validation_data=val_data, epochs=15, callbacks=[ch, es, re_lr])
        return history

    def predict_evaluate(self,test_data, val_data):
        y_true = test_data.classes
        y_pred = np.argmax(self.model.predict(test_data), axis=1)
        print("Cnn model performance")
        cnn_ac = accuracy_score(y_true, y_pred)
        cnn_pr = precision_score(y_true, y_pred, average='weighted')
        cnn_re = recall_score(y_true, y_pred, average='weighted')
        cnn_f1 = f1_score(y_true, y_pred, average='weighted')
        print("cnn accuracy:",cnn_ac)
        print("cnn precision score:",cnn_pr)
        print("cnn recall_score:",cnn_re)
        print("cnn f1 score:",cnn_f1)

        return cnn_ac, cnn_pr, cnn_re,cnn_f1


    def cnn_plot(self,history):
        """
        creating a plots training and validation accuracy and loss for each epoch
        """
        acc= history.history['accuracy']
        val_ac = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        #plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_ac, 'ro-', label='Validation Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title(f'cnn model  accuracy per epoch')
        plt.legend()

        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'cnn model loss per epoch')
        plt.legend()

        plt.show()

class VGG19Model:
    def __init__(self):
        base_model = VGG19(input_shape=(128,128,3), include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        
        x = Flatten()(base_model.output)
        x = Dense(4608, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1152, activation='relu')(x)
        output = Dense(4, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    
    def train_and_evaluate(self, train_data, val_data):
        model_save_path = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\model\VGG19model.keras"
    
        cp = ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
        esp = EarlyStopping(monitor="val_loss", patience=3, verbose=1, restore_best_weights=True)  # Reduced patience
        relr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)  # Monitor val_loss

        history = self.model.fit(train_data, validation_data=val_data, epochs=15, callbacks=[cp, esp, relr])  # Adjusted epochs

        return history

    def predict_evaluate(self,test_data, val_data):
        y_true = test_data.classes
        y_pred = np.argmax(self.model.predict(test_data), axis=1)

        vg_ac = accuracy_score(y_true, y_pred)
        vg_pr = precision_score(y_true, y_pred, average='weighted')
        vg_re = recall_score(y_true, y_pred, average='weighted')
        vg_f1 = f1_score(y_true, y_pred, average='weighted')
        print("vgg19 accuracy:",vg_ac)
        print("vgg19 precision score:",vg_pr)
        print("vgg19 recall_score:",vg_re)
        print("vgg19 f1 score:",vg_f1)

        return vg_ac, vg_pr, vg_re,vg_f1

    def vgg_plot(self,history):
        """
        creating a plots training and validation accuracy and loss for each epoch
        """
        acc= history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        #plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title(f'Vgg19 accuracy per epoch')
        plt.legend()

        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'VGG19 model loss per epoch')
        plt.legend()

        plt.show()

class ModelCompare:
    def __init__(self, models, val_data):
        self.models = models  
        self.val_data = val_data 
    
    def compare_models(self, vg_ac, vg_pr, vg_re, vg_f1, cnn_ac, cnn_pr, cnn_re, cnn_f1):
        acc = {'CNN': cnn_ac, 'VGG19': vg_ac}
        best_model_name = max(acc, key=acc.get) 

        print(f"\nThe best model is {best_model_name} with an accuracy of {acc[best_model_name]:.4f}")

       
        best_model = self.models["custom_cnn"] if best_model_name == "CNN" else self.models["vgg19"]

        
        model_save_path = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\model\bestmodel.keras"
        best_model.model.save(model_save_path)  
        print(f"Best model saved as '{model_save_path}'")

        
        