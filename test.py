import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier
from RentHop import RentHop
#

train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")
print("Read files")
rhop = RentHop()



train_X, train_Y ,test_X = rhop.getData(train_df,test_df)

x_train, x_test, y_train, y_test = train_test_split(train_X,train_Y,test_size=0.3)

model = GradientBoostingClassifier(n_estimators=1500)

model.fit(x_train,y_train)
y_pred = model.predict_proba(x_test)
print(log_loss(y_test,y_pred))
exit()
# model.fit(train_X,train_Y)
pred = model.predict_proba(test_X)

res = pd.DataFrame(pred,columns = ['high','medium','low'],index=test_df.listing_id)
res.to_csv("files/gradientbx.csv")


