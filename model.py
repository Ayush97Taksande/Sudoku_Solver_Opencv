import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

path = 'Digits_new'
myList = os.listdir(path)

images = []
classNo = []
numOfClasses = len(myList)

# Load images from the dataset
for x in range(numOfClasses):
    myPiclist = os.listdir(os.path.join(path, str(x)))
    for i, y in enumerate(myPiclist):
        curImg = cv2.imread(os.path.join(path, str(x), y))
        if curImg is not None:
            curImg = cv2.resize(curImg, (32, 32))
            images.append(curImg)
            classNo.append(x)
        else:
            print(f"Error reading image: {os.path.join(path, str(x), y)}")
    print(x, end=' ')
print(" ")
images=np.array(images)
classNo=np.array(classNo)

# Splitting the data
test_ratio = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=test_ratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

# Preprocessing function
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# Apply preprocessing to images
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# Reshape data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Image augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

# One-hot encoding of labels
Y_train = to_categorical(Y_train, numOfClasses)
Y_test = to_categorical(Y_test, numOfClasses)
Y_validation = to_categorical(Y_validation, numOfClasses)

# Model architecture
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=sizeOfPool))
    model.add(tf.keras.layers.Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(tf.keras.layers.Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=sizeOfPool))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(noOfNodes, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(numOfClasses, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# Train the model
history = model.fit(dataGen.flow(X_train, Y_train, batch_size=50), steps_per_epoch=2000, epochs=10, validation_data=(X_validation, Y_validation), shuffle=1)

# Plot training & validation loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score =', score[0])
print('Test Accuracy =', score[1])

# Save model using Keras
model.save("model_trained_new3h5")  # Save in Keras format

# Load model (for making predictions later)
# from tensorflow.keras.models import load_model
# model = load_model("model_trained.h5")

# Use the model for predictions on a new image
def predict_image(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    prediction = model.predict(img)
    return np.argmax(prediction)

# Example usage
# predicted_class = predict_image('path_to_image.jpg', model)
# print("Predicted class:", predicted_class)
