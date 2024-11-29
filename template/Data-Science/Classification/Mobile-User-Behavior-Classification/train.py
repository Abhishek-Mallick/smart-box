import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ANN():
    def load_data(self, data_path):
        data_path = data_path
        self.data = pd.read_csv(data_path)
    
    def one_hot_encoding(self):
        del self.data['User ID']
        categorical_cols = []
        for col in self.data.columns:
            if self.data[col].dtypes == 'O':
                categorical_cols.append(col)
        
        data_encoded = pd.get_dummies(self.data, columns = categorical_cols, drop_first = True)
        return data_encoded

    def split_data(self, data_encoded):
        X = data_encoded.drop('User Behavior Class', axis = 1)
        y = data_encoded['User Behavior Class']

        return train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    def data_transformer(self, X_train, X_test):
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

    def build_network(self):
        model = Sequential([
        Dense(12, activation = 'relu', input_dim = self.X_train_scaled.shape[1]),
        Dense(12, activation = 'relu'),
        Dense(5, activation = 'softmax')
        ])
        return model
    
    def train_model(self, model, y_train):
        history = model.fit(self.X_train_scaled, y_train - 1, batch_size = 32, epochs = 100, validation_split = 0.2, verbose = 1)
        return history

    def show_plot(self, history):
        plt.figure(figsize = (10, 6))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label = 'Training Accuracy')
        plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label = 'Training Loss')
        plt.plot(history.history['val_loss'], label = 'Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = np.argmax(model.predict(X_test), axis = 1) + 1
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def save_model(self, model, model_path, scaler_path):
        model.save(model_path)
        file = open(scaler_path, 'wb')
        pickle.dump(self.scaler, file)
        print(f"Model saved to {model_path}")


folder = './data'
filename = 'user_behavior_dataset.csv'
full_path = os.path.join(folder, filename)
# print(full_path)

model_folder = './model'
model_filename = 'user_behavior_model.keras'
model_path = os.path.join(model_folder, model_filename)

scaler_filename = 'scaler.pkl'
scaler_path = os.path.join(model_folder, scaler_filename)

if __name__== "__main__":
    ann = ANN()
    ann.load_data(full_path)

    # One hot encoding 
    data_encoded = ann.one_hot_encoding()

    # Split the train and test data 
    X_train, X_test, y_train, y_test = ann.split_data(data_encoded)

    # Transform the data 
    ann.data_transformer(X_train, X_test)

    # Build and compile the model 
    model = ann.build_network()
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.01), metrics = ['accuracy'])

    # Train the model 
    print('Training start...')
    history = ann.train_model(model, y_train)
    print('Training end')
    
    # Show the training and validation plot 
    ann.show_plot(history)

    # Evaluate the model
    ann.evaluate_model(model, ann.X_test_scaled, y_test)

    # Save model 
    ann.save_model(model, model_path, scaler_path)



