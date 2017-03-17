# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import numpy as np
import pandas as pd
from RentHop import RentHop
from sklearn.model_selection import train_test_split

data_dim = 31
nb_classes = 3

train_df = pd.read_json("input/train.json")
train_df = train_df
test_df = pd.read_json("input/test.json")
print("Read files")
rhop = RentHop()

train_X, train_Y = rhop.getTrainNet(train_df)

test_X = rhop.getTestNet(test_df)

model = Sequential()
model.add(Dense(500, input_dim=np.shape(train_X)[1], init='uniform', activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(PReLU())

model.add(Dense(100, init='uniform', activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(PReLU())

model.add(Dense(3, init='uniform', activation='softmax'))
#
model.compile(loss='categorical_crossentropy',
           optimizer='adam')
model.fit(train_X, train_Y,
          nb_epoch=20,
          batch_size=2000,
          validation_split=0.33)

#score = model.evaluate(train_X, train_Y, batch_size=16)
pred = model.predict_proba(test_X)


res = pd.DataFrame(pred,columns = ['high','medium','low'],index=test_df.listing_id)
res.to_csv("files/net.csv")
