import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time

model=load_model(r"C:\Users\ASUS\Desktop\problem c\model04_10_0.10.h5")   

# DIR=r"C:/Users/ASUS/Desktop/problem c/Asian_giant_hornets_1.jpg"
# DIR=r"C:/Users/ASUS/Desktop/problem c/Asian_giant_hornets_2.jpg"
# DIR=r"C:/Users/ASUS/Desktop/problem c/Asian_giant_hornets_3.jpg"

# DIR=r"C:/Users/ASUS/Desktop/problem c/Cicada killer.jpg"
# DIR=r"C:/Users/ASUS/Desktop/problem c/European_hornet.jpg"
# DIR=r"C:/Users/ASUS/Desktop/problem c/ichneumon.jpg"
# DIR=r"C:/Users/ASUS/Desktop/problem c/Vespidae.jpg"
DIR=r"C:/Users/ASUS/Desktop/problem c/Bombus Spp.jpg"
IMG_SIZE = 128

img_array= cv2.imread(DIR) 

image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

prob = model.predict(image)[0]

print("Negative ID Probability: ", prob[1] * 100)
print("Predictions: ", prob)

prob_show = "Negative ID Probability: " + str(prob[1] * 100)

cv2.putText(img_array, prob_show, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

cv2.imshow("input_image", img_array)
# cv2.waitKey(0)
key = cv2.waitKey(10)
if key == 27: # exit on ESC
    cv2.destroyAllWindows() 
