import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,VotingClassifier
import xgboost as xgb
from RentHop import RentHop


train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")

print("Read files")
rhop = RentHop()

train_X, train_Y = rhop.getTrain(train_df)
test_X = rhop.getTest(test_df)
target_num_map = {'high':0, 'medium':1, 'low':2}


train_Y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

x_train, x_test, y_train, y_test = train_test_split(train_X,train_Y,test_size=0.3)


extraT = ExtraTreesClassifier(n_estimators=1000)
# extraT.fit(x_train,y_train)
#print(log_loss(y_test,extraT.predict_proba(x_test)))

rf1 = RandomForestClassifier(n_estimators=1000,criterion='gini')
rf2 = RandomForestClassifier(n_estimators=1000,criterion='entropy')
rf3 = RandomForestClassifier(n_estimators=1000,criterion='gini',max_features="log2")
# rf.fit(x_train,y_train)
# print(log_loss(y_test,rf.predict_proba(x_test)))

gra = GradientBoostingClassifier(n_estimators=1000)
gra.fit(x_train,y_train)
print(log_loss(y_test,gra.predict_proba(x_test)))
exit()
b = xgb.XGBClassifier(n_estimators=200)

model = VotingClassifier(estimators=[('rf',rf1),('rf2',rf2),('rf3',rf3) ,('gra',gra),("xgb",b)],voting='soft')
model.fit(x_train,y_train)
print(log_loss(y_test,model.predict_proba(x_test)))



pred = model.predict_proba(test_X)

res = pd.DataFrame(pred,columns = ['high','medium','low'],index=test_df.listing_id)
res.to_csv("files/voting.csv")


