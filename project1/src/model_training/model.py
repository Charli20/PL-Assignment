import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Flatten, LeakyReLU, GRU, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from src.config import Config
from tensorflow.keras.regularizers import l2

class DNNModel:
    @staticmethod
    def model_train(x_train, x_test, y_train, y_test, config):
        save_path = config.MODEL_SAVE_PATH
        os.makedirs(save_path, exist_ok=True)

        
        dnn = Sequential([
            Input(shape=(x_train.shape[1],)), 
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),  
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(41, activation='softmax')  
        ])
            
        
        dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        
        best_model_path = os.path.join(save_path, 'best_dnn_model.keras')
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

   
        dnn.fit(x_train, y_train, epochs=15, batch_size=64, verbose=1,
                      validation_data=(x_test, y_test), callbacks=[checkpoint, early_stop])

        best_dnn_model = load_model(best_model_path)

       
        y_pred_dnn = np.argmax(best_dnn_model.predict(x_test), axis=-1) 

        dnn_ac = accuracy_score(y_test, y_pred_dnn)
        dnn_ps = precision_score(y_test, y_pred_dnn, average='weighted')
        dnn_re = recall_score(y_test, y_pred_dnn, average='weighted')
        dnn_f1 = f1_score(y_test, y_pred_dnn, average='weighted')
        dnn_mse = mean_squared_error(y_test, y_pred_dnn)
        dnn_r2 = r2_score(y_test, y_pred_dnn)

        print("DNN model performance:")
        print(f"Dnn accuracy: {dnn_ac:.4f}")
        print(f"Dnn precision score: {dnn_ps:.4f}")
        print(f"Dnn recall score: {dnn_re:.4f}")
        print(f"Dnn F1 score: {dnn_f1:.4f}")
        print(f"Dnn model training is completed and the best model is saved as '{best_model_path}'.")

        return best_dnn_model, y_pred_dnn, dnn_ac, dnn_ps, dnn_re, dnn_f1

class CNNModel:
    @staticmethod
    def model_train(x_train, x_test, y_train, y_test, config, input_shape):
        save_path = config.MODEL_SAVE_PATH
        os.makedirs(save_path, exist_ok=True)

        x_train = x_train.to_numpy() 
        x_test = x_test.to_numpy()   

        x_train = x_train.reshape(-1, input_shape, 1) 
        x_test = x_test.reshape(-1, input_shape, 1)

        cnn = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
            MaxPooling1D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(41, activation='softmax')
        ])

        cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        best_model_path = os.path.join(save_path, 'best_cnn_model.keras')

        cp = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        cnn.fit(x_train, y_train, epochs=15, batch_size=64, verbose=1, validation_data=(x_test, y_test), callbacks=[cp, es])

        best_cnn_model = load_model(best_model_path)
        y_pred_cnn = np.argmax(best_cnn_model.predict(x_test), axis=-1)

        cnn_ac = accuracy_score(y_test, y_pred_cnn)
        cnn_re = recall_score(y_test, y_pred_cnn, average='weighted')
        cnn_f1 = f1_score(y_test, y_pred_cnn, average='weighted')
        cnn_ps = precision_score(y_test, y_pred_cnn, average='weighted')

        print("CNN model performance:")
        print(f"Cnn accuracy: {cnn_ac:.4f}")
        print(f"Cnn precision score: {cnn_ps:.4f}")
        print(f"Cnn recall score: {cnn_re:.4f}")
        print(f"Cnn F1 Score: {cnn_f1:.4f}")
        print(f"Cnn model training is completed and the best model is saved as '{best_model_path}'.")

        return best_cnn_model, y_pred_cnn, cnn_ac, cnn_ps, cnn_re, cnn_f1

class GRUModel:
    @staticmethod
    def model_train(x_train, x_test, y_train, y_test, config, input_shape):
        save_path = config.MODEL_SAVE_PATH
        os.makedirs(save_path, exist_ok=True)

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

        x_train = x_train.reshape(-1, input_shape, 1)  
        x_test = x_test.reshape(-1, input_shape, 1)

        gru = Sequential([
            GRU(128, return_sequences=True, input_shape=(input_shape, 1)),
            GRU(64),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(41, activation='softmax')
        ])

        gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        best_model_path = os.path.join(save_path, 'best_gru_model.keras')

        cp = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        gru.fit(x_train, y_train, epochs=15, batch_size=64, verbose=1, validation_data=(x_test, y_test), callbacks=[cp, es])

        best_gru_model = load_model(best_model_path)
        y_pred_gru = np.argmax(best_gru_model.predict(x_test), axis=-1)

        gru_ac = accuracy_score(y_test, y_pred_gru)
        gru_ps = precision_score(y_test, y_pred_gru, average='weighted')
        gru_re = recall_score(y_test, y_pred_gru, average='weighted')
        gru_f1 = f1_score(y_test, y_pred_gru, average='weighted')

        print("GRU model performance:")
        print(f"GRU accuracy: {gru_ac:.4f}")
        print(f"GRU precision score: {gru_ps:.4f}")
        print(f"GRU recall score: {gru_re:.4f}")
        print(f"GRU F1 Score: {gru_f1:.4f}")

        print(f"GRU model training is completed and the best model is saved as '{best_model_path}'.")

        return best_gru_model, y_pred_gru, gru_ac, gru_ps, gru_re, gru_f1
 
class ModelCompair:
    def __init__(self, dnn_model, cnn_model, gru_model, save_path):
        self.dnn_model = dnn_model
        self.cnn_model = cnn_model
        self.gru_model = gru_model
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def compare_models(self, dnn_acc, cnn_acc, gru_acc):
        """
        Compare the DNN, CNN, and GRU models 
        """
        acc = {'DNN': dnn_acc,'CNN': cnn_acc,'GRU': gru_acc}
        
        best_model_name = max(acc, key=acc.get)
        print(f"\nThe best model is {best_model_name} with an accuracy of {acc[best_model_name]:.4f}")

        best_model_path = os.path.join(self.save_path, 'best_model.keras')

        if best_model_name == 'DNN':
            self.dnn_model.save(best_model_path)
        elif best_model_name == 'CNN':
            self.cnn_model.save(best_model_path)
        elif best_model_name == 'GRU':
            self.gru_model.save(best_model_path)

        print(f"Best model saved as: {best_model_path}")
        return best_model_name, best_model_path