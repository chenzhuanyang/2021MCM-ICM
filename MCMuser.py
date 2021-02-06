import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time

model=load_model("C:/Users/ASUS/Desktop/problem c/model03_59_0.02.h5")
IMG_SIZE = 64

DIR="C:/Users/ASUS/Desktop/problem c/is01.jpg"
# DIR="C:/Users/ASUS/Desktop/problem c/is02.jpeg"
# DIR="C:/Users/ASUS/Desktop/problem c/no01.jpeg"

img_array= cv2.imread(DIR)

image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

prob = model.predict(image)[0]

print("Negative ID Probability: ", prob[1] * 100)
print("Predictions: ", prob)

prob_show = "Negative ID Probability: " + str(prob[1] * 100)

cv2.putText(img_array, prob_show, (10, 25),  
            cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

cv2.imshow("input_image", img_array)
key = cv2.waitKey(10)
if key == 27: # exit on ESC
    cv2.destroyAllWindows() 