import pandas as pd
import numpy as np
from sklearn import preprocessing

class RentHop:

    def __init__(self):
        self.features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price","num_photos","num_features",\
                    "created_year","created_month","created_day","created_hour","feat_elevator","feat_animals_allowed","feat_hardwood_floor",\
                    "feat_doorman","feat_dishwasher","feat_no_fee","feat_laundry_unit","feat_laundry_building",\
                    "feat_fit_center","feat_pre_war","feat_roof_deck","feat_outdoor_space","feat_pool","feat_new_construction",\
                    "feat_terrace","feat_loft","washer","parking","internet","distance",'bed_bath_sum',"bed_bath_diff","manager_id",\
                    "building_id","listing_id"]
        self.target_num_map = {'high':0, 'medium':1, 'low':2}

    def getData(self,train,test):
        x_train,y_train = self.getTrain(train)
        x_test= self.getTest(test)

        self.addCommonFeature(x_train,x_test,'manager_id')
        self.addCommonFeature(x_train, x_test,'building_id')
        self.addCommonFeature(x_train, x_test, 'listing_id')

        return (x_train.as_matrix(),y_train,x_test.as_matrix())

    def getDataNet(self,train,test):
        pass

    def getTrain(self,train):
        df = train.copy()
        self.addFeatures(df)
        train_X = df[self.features_to_use]
        train_Y = np.array(df['interest_level'].apply(lambda x: self.target_num_map[x]))

        return (train_X,train_Y)

    def getTrainNet(self, train):
        train_X,train_Y = self.getTrain(train)
        train_X = preprocessing.StandardScaler().fit_transform(train_X)
        train_Y = pd.get_dummies(train_Y).values

        return (train_X,train_Y)

    def addFeatures(self, df):

        df.loc[df['bedrooms'] > 6, 'bedrooms'] = 6
        df.loc[df['bathrooms'] > 6, 'bathrooms'] = 6
        # df.loc[df['bathrooms'] == 0, 'bathrooms'] = 1
        # df.loc[df['bedrooms'] == 0, 'bedrooms'] = 1

        df["num_photos"] = df["photos"].apply(len)
        df["num_features"] = df["features"].apply(len)
        df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
        df["created"] = pd.to_datetime(df["created"])
        df["created_year"] = df["created"].dt.year
        df["created_month"] = df["created"].dt.month
        df["created_day"] = df["created"].dt.day
        df["created_hour"] = df["created"].dt.hour
        # df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']
        df['feat_elevator'] = df['features'].map(lambda x: 'Elevator' in x)
        df['feat_animals_allowed'] = df['features'].map(lambda x: ('Cats Allowed' in x) or ('Dogs Allowed' in x))
        df['feat_hardwood_floor'] = df['features'].map(lambda x: ('Hardwood Floors' in x) or ('HARDWOOD' in x))
        df['feat_doorman'] = df['features'].map(lambda x: 'Doorman' in x)
        df['feat_dishwasher'] = df['features'].map(lambda x: 'Dishwasher' in x)
        df['feat_no_fee'] = df['features'].map(lambda x: 'No Fee' in x)
        df['feat_laundry_unit'] = df['features'].map(lambda x: ('Laundry in Unit' in x))
        df['feat_laundry_building'] = df['features'].map(lambda x: ('Laundry in Building' in x))
        df['feat_fit_center'] = df['features'].map(lambda x: 'Fitness Center' in x)
        df['feat_pre_war'] = df['features'].map(lambda x: ('Pre-War' in x) or ('prewar' in x))
        df['feat_roof_deck'] = df['features'].map(lambda x: 'Roof Deck' in x)
        df['feat_outdoor_space'] = df['features'].map(lambda x: ('Outdoor Space' in x) or ('Common Outdoor Space' in x))
        df['feat_pool'] = df['features'].map(lambda x: 'Swimming Pool' in x)
        df['feat_new_construction'] = df['features'].map(lambda x: 'New Construction' in x)
        df['feat_terrace'] = df['features'].map(lambda x: 'Terrace' in x)
        df['feat_loft'] = df['features'].map(lambda x: 'Loft' in x)
        df['washer'] = df['features'].map(lambda x: ('washer' in x) or ('Washer' in x))
        df['parking'] = df['features'].map(lambda x: ('parking' in x) or ('Parking' in x))
        df['internet'] = df['features'].map(lambda x: ('internet' in x) or ('Internet' in x))
        df['bed_bath_diff'] = df['bedrooms'] - df['bathrooms']
        df['bed_bath_sum'] = df['bedrooms'] + df['bathrooms']

        def pointTopoint(a, b):
            a = (np.radians(a[0]), np.radians(a[1]))
            b = (np.radians(b[0]), np.radians(b[1]))

            dlon = b[1] - a[1]
            dlat = b[0] - a[0]
            a = np.power(np.sin(dlat / 2), 2) + np.cos(a[0]) * np.cos(b[0]) * np.power((np.sin(dlon / 2)), 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = 3961 * c
            return d
        bronx_center = (40.834600, -73.879082)
        df['distance'] = pointTopoint((df.latitude, df.longitude), bronx_center)

        # df['price_per_bed'] = df['price']/df['bedrooms']
        # df['price_per_bath'] = df['price'] / df['bathrooms']
        # df['price_per_room'] = df['price'] / (df['bathrooms'] + df['bedrooms'])

    def addCommonFeature(self,x_train,x_test, feature_name):
        # manager id
        feature_list = list(x_train[feature_name].values) + list(x_test[feature_name].values)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(feature_list)
        x_train[feature_name] = lbl.transform(x_train[feature_name])
        x_test[feature_name] = lbl.transform(x_test[feature_name])



    def getTest(self,test):
        df = test.copy()
        self.addFeatures(df)
        test_X = df[self.features_to_use]

        return test_X

    def getTestNet(self,test):
        df = test.copy()
        self.addFeatures(df)

        test_X = df[self.features_to_use].as_matrix()
        test_X = preprocessing.StandardScaler().fit_transform(test_X)
        return test_X

