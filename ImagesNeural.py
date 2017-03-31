import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from RentHop import RentHop
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from RentHop import RentHop
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import random

train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")

listing_id = list(train_df["listing_id"])

X = []
Y = []

model = VGG16(weights='imagenet', include_top=False)

for row in train_df.iterrows():
    l = row[1]["listing_id"]
    interest_level = row[1]["interest_level"]

    if os.path.isdir("../images_sample//{0}".format(l)):
        for img in os.listdir("../images_sample/{0}".format(l)):
            img_path = "../images_sample/{0}/{1}".format(l,img)
            img = image.load_img(img_path, target_size=(224, 224))
            x = preprocess_input(np.array(img_for_clf.convert("RGB")).reshape(1, 224, 224, 3).astype(np.float64))
            features = model.predict(x)

            print(np.shape(features))
            break

            X.append(np.asarray(Image.open("../images_sample/{0}/{1}".format(l,img)).convert('L')).flatten())
            Y.append(interest_level)



#X = np.reshape(X,(167,1))

#X = np.reshape(X,(167,27200))




img_path = 'elephant.jpg'

