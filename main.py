import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Step 1: Load the model using pickle
model_path = 'model_trained.p'  # Update with your actual model path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Step 2: Preprocess the image
img_path = 'img004-00001.png'  # Update with your image path
img = image.load_img(img_path, target_size=(224, 224))  # Adjust size based on your model's input requirement
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# If the model requires normalization or preprocessing (adjust as needed)
# Example: if the model was trained with normalized inputs (0-1 range)
img_array = img_array / 255.0  # Normalize the image

# Step 3: Make predictions using the model
predictions = model.predict(img_array)  # Assuming the model has a predict method
print("Predictions:", predictions)

# Step 4: Interpret predictions
# This depends on your model type (classification or regression)
# Example: if it's a classification model, the result might be probabilities for each class
predicted_class = np.argmax(predictions, axis=1)
print("Predicted Class:", predicted_class)
