import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
import dclab
import pandas as pd

# Setting up constants
IMG_SIZE = (250,60)
EPOCHS =20


# Load trainning set
# x - images , y - class identifier ['doublet', 'singlet'] 
x, y = [], []

# Iterate through each trainning datasets folder in path
train_path = 'F://RTDC//train_set'
cls_count = 0
for train_class in os.listdir(train_path):
    for file in os.listdir(os.path.join(train_path, train_class)):
        y += [cls_count]
        x += [cv2.resize(cv2.imread(os.path.join(train_path, train_class, file)),tuple(reversed(IMG_SIZE)))]
    cls_count += 1

# Convert to numpy array
x = np.array(x)
y = np.eye(cls_count)[y]


# Initializing the model
model = keras.Sequential([
    keras.layers.Conv2D(32,3, input_shape = (250, 60, 3), activation = 'relu'),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.MaxPool2D(strides=2),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.MaxPool2D(strides=2),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.MaxPool2D(strides=2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])


# Initializing the adam optimizer
Optimizer = keras.optimizers.Adam
Opt=Optimizer(lr=1e-3, decay=1e-3 / EPOCHS)

# Compiling the model
# after trial and error with different loss functions, this one had the optimal result
model.compile(optimizer=Opt, 
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x,y,epochs=EPOCHS)
print('End of Training')


#Load testing set

"""
# extract 900 images and write to local path using index as name"
ds = dclab.new_dataset("F://RTDC//Data001.rtdc")
imgarray = ds["image"]

for idx in range(100,1000):
    filename += [idx]
    cv2.imwrite('F://RTDC//test_set_from_rtdc//'+ str(idx) +'.png', imgarray[idx])  
"""

x_test, filename = [], []

# iterate through selected test images


test_path = 'F://RTDC//test_set_from_rtdc'
# Iterate through each image-file in path
for file in os.listdir(test_path):
        filename += [file]
        x_test += [cv2.resize(cv2.imread(os.path.join(test_path, file)),tuple(reversed(IMG_SIZE)))]


# Convert to numpy array
x_test = np.array(x_test)


# predicting resuls
predictions = model.predict(x_test)
predicted_cls = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1)

#printing class predictions 
predictions

#displaying as a pandas df where 0:doublet,1:singlet
pd.set_option('display.max_rows', 50)
df = pd.DataFrame({'Actual':filename,
 'Predicted_class':predicted_cls,
 'confidence':confidence
})

df
gfg_csv_data = df.to_csv('F://RTDC//test_set_from_rtdc//predict_result.csv', index = True) 
