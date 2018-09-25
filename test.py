from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
file_path = "./upload/Hydrangeas.jpg"
data = load_img(file_path, target_size=(224,224))
data = img_to_array(data)
data = np.expand_dims(data, axis = 0)
data = preprocess_input(data)

model = ResNet50(weights="imagenet")
preds = model.predict(data)
print(preds)
print(decode_predictions(preds, top=3)[0])
