#import packages
import tensorflow
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil
import os
import pandas as pd
from ast import literal_eval

from scipy.io import loadmat

from tensorflow.python.client import device_lib
import wandb
from wandb.keras import WandbCallback

numEpochs = 25
batchSize = 32
wandb.init(project="my-test-project", entity="grav17")
wandb.config = {
  "learning_rate": 1e-4,
  "epochs": numEpochs,
  "batch_size":batchSize 
}

# ... Define a model

print(device_lib.list_local_devices()) # list of DeviceAttributes


# box_coord = []
# obj_contour = []
# for i,j,all_files in os.walk('caltech-101\caltech-101\Annotations\Airplanes_Side_2'):
#     for f in all_files:
#         data_dict = loadmat(os.path.join(i,f))
#         box_coord.append((data_dict['box_coord'],data_dict['obj_contour']))

# df = pd.DataFrame(box_coord)
# df.columns = ['box_coord','obj_contour']
# print(df.head())
# df.to_csv('true_labels.csv',sep=',')



# data_dict = loadmat('caltech-101/caltech-101/Annotations/Airplanes_Side_2/annotation_0001.mat')
# data_array = {"box_coord": data_dict['box_coord'], "obj_contour": data_dict['obj_contour']}
# # data_array = data_array.transpose(1, 0)
# df = pd.DataFrame([data_array])
# print(df.head())

df = pd.read_csv('true_labels.csv')
# print(df.info())
i=0
image_nums = ['image_' + str(i) for i in range(1,801)] #y1,y2,x1,x2
df['img_name'] = image_nums
# print(df.head())
box_coord = df['box_coord'].tolist()

data = []
targets = []
filenames = []

for num,i in enumerate(box_coord):
    b = i.strip('][').split(' ')
    b=[j for j in b if j]
    y1, y2, x1, x2 = b
    y1=int(y1)
    y2=int(y2)
    x1=int(x1)
    x2=int(x2)
    image_path = os.path.join('dataset/airplanes','image_'+str(f"{num+1:04d}")+'.jpg')
    sample_img = cv2.imread(image_path)
    (h, w) = sample_img.shape[:2]
    #Normalization
    x1 =float(x1)/w
    x2 =float(x2)/w 
    y1 =float(y1)/w 
    y2 =float(y2)/w
    # print(x1,x2,y1,y2)
    image=load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    data.append(image)
    targets.append((x1,y1,x2,y2))
    filenames.append('image_'+str(f"{num+1:04d}")+'.jpg')


data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
split = train_test_split(data, targets, filenames, test_size=0.10,random_state=42)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

opt = Adam(lr=1e-4)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")

H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=batchSize,
	epochs=numEpochs,
	verbose=1,
    callbacks=[WandbCallback()])

print("[INFO] saving object detector model...")
model.save("output/myModel.h5", save_format="h5")
# plot the model training history
N=numEpochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("output/myPlot.png")