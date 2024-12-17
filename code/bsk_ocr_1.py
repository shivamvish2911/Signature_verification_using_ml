import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import _KerasLazyLoader
from keras import layers, models

train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

X_train /= 255.0
X_test /= 255.0

num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

def build_ocr_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

ocr_model = build_ocr_model()
ocr_model.summary()

ocr_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
  
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 10:  
            char_image = binary_image[y:y+h, x:x+w]
            char_image = cv2.resize(char_image, (28, 28))  
            character_images.append(char_image)
    
    return character_images

def recognize_characters(character_images):
    predictions = []
    for char_img in character_images:
       
        char_img = char_img.reshape(1, 28, 28, 1)
        char_img = char_img.astype('float32') / 255.0

        pred = ocr_model.predict(char_img)
        predicted_class = np.argmax(pred)
        predictions.append(predicted_class)
    
    return predictions

ocr_model.save('ocr_model.h5')

print("sucess")