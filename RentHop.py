import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from itertools import product
from sklearn.preprocessing import MultiLabelBinarizer
import re
from nltk.stem import PorterStemmer

class RentHop:

    def __init__(self):
        self.features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price","num_photos","num_features",\
                    "created_year","created_month","created_day","created_hour","feat_elevator","feat_animals_allowed","feat_hardwood_floor",\
                    "feat_doorman","feat_dishwasher","feat_no_fee","feat_laundry_unit","feat_laundry_building",\
                    "feat_fit_center","feat_pre_war","feat_roof_deck","feat_outdoor_space","feat_pool","feat_new_construction",\
                    "feat_terrace","feat_loft","washer","parking","internet","distance1","distance2","distance3","distance4",\
                                 'bed_bath_sum',"bed_bath_diff","manager_id","building_id","listing_id", "price_per_bed",\
                                 "price_per_bath","price_per_room",\
                    "comp_ppr_d1","comp_ppr_d2","comp_ppr_d3"] +\
                    ['bed_bath', 'bath_bed', 'bath_bed_norm', 'bed_bath_norm']
        self.target_num_map = {'high':0, 'medium':1, 'low':2}

    def getData(self,train,test, decomposition={"type":"None","value":0}):
        train = train.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})

        train_X = self.addFeatures(train)
        test_X = self.addFeatures(test)

        train_X, test_X = self._hcce(train_X,test_X)
        train_X, test_X = self._binarize(train_X, test_X)


        # if decomposition["type"] == 'PCA':
        #     pca = PCA(decomposition["value"])
        #     pca.fit(x_train, x_test)
        #     x_train = pca.transform(x_train)
        #     x_test = pca.transform(x_test)

        train_Y = train_X["interest_level"]

        train_X.drop("interest_level",axis=1,inplace = True)

        for fea in ["photos","features","created","description",'building_id', 'manager_id','display_address', 'street_address']:
            train_X.drop(fea,axis=1,inplace=True)
            test_X.drop(fea, axis=1, inplace=True)

        if type(train_X) is not np.ndarray:
            train_X = train_X.as_matrix()
            test_X = test_X.as_matrix()

        return (train_X,train_Y,test_X)

    def getDataNet(self,train,test):
        pass

    def addFeatures(self, df):

        df.loc[df['bedrooms'] > 6, 'bedrooms'] = 6
        df.loc[df['bathrooms'] > 6, 'bathrooms'] = 6

        fmt = lambda s: s.replace("\u00a0", "").strip().lower()

        df["street_address"] = df['street_address'].apply(fmt)
        df["display_address"] = df["display_address"].apply(fmt)
        df["desc_wordcount"] = df["description"].apply(len)
        df["num_photos"] = df["photos"].apply(len)
        df["num_features"] = df["features"].apply(len)
        df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
        df["created"] = pd.to_datetime(df["created"])
        df["created_year"] = df["created"].dt.year
        df["created_month"] = df["created"].dt.month
        df["created_day"] = df["created"].dt.day
        df["created_hour"] = df["created"].dt.hour

        df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']
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
        df['price_per_bed'] = df['price'] / df['bedrooms']
        df['price_per_bath'] = df['price'] / df['bathrooms']
        df['price_per_room'] = df['price'] / (df['bathrooms'] + df['bedrooms'])
        df["bedsPerc"] = df["bedrooms"] / (df['bedrooms'] + df['bathrooms'])
        df['bed_bath'] = df['bedrooms'] / (df['bathrooms'])
        df['bath_bed'] = (df['bathrooms'])/df['bedrooms']
        df['bath_bed_norm'] = (df['bathrooms']) / (df['bedrooms'] + df['bathrooms'])
        df['bed_bath_norm'] = (df['bedrooms']) / (df['bedrooms'] + df['bathrooms'])

        def pointTopoint(a, b):
            a = (np.radians(a[0]), np.radians(a[1]))
            b = (np.radians(b[0]), np.radians(b[1]))

            dlon = b[1] - a[1]
            dlat = b[0] - a[0]
            a = np.power(np.sin(dlat / 2), 2) + np.cos(a[0]) * np.cos(b[0]) * np.power((np.sin(dlon / 2)), 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = 3961 * c
            return d

        df['distance1'] = pointTopoint((df.latitude, df.longitude), (40.780181, -73.969279))#center manhatan
        df['distance2'] = pointTopoint((df.latitude, df.longitude), (40.734085, -73.995801))# center sout manhatan
        df['distance3'] = pointTopoint((df.latitude, df.longitude), (40.826765, -73.916665))  # center bronx
        df['distance3'] = pointTopoint((df.latitude, df.longitude), (40.740979, -73.920613))  # sunnyside
        df['distance4'] = pointTopoint((df.latitude, df.longitude), (40.682555, -73.947907))  # brookling center
        df['distance5'] = pointTopoint((df.latitude, df.longitude), (40.715612, -73.839417))  # forest hills
        df['distance6'] = pointTopoint((df.latitude, df.longitude), (40.825336, -73.952027))  # upper manh
        df['distance7'] = pointTopoint((df.latitude, df.longitude), (40.710928, -74.010392))  # lower manh

        df['comp_ppr_d1'] = df['price_per_room'] * 100*df['distance1']
        df['comp_ppr_d2'] = df['price_per_room'] * 100*df['distance2']
        df['comp_ppr_d3'] = df['price_per_room'] * 100*df['distance3']

        df.fillna(-1, inplace=True)
        df.replace(np.inf,-1,inplace=True)

        return df

    def add_featuresBrandon(self,df):
        fmt = lambda s: s.replace("\u00a0", "").strip().lower()
        df["photo_count"] = df["photos"].apply(len)
        df["street_address"] = df['street_address'].apply(fmt)
        df["display_address"] = df["display_address"].apply(fmt)
        df["desc_wordcount"] = df["description"].apply(len)
        df["pricePerBed"] = df['price'] / df['bedrooms']
        df["pricePerBath"] = df['price'] / df['bathrooms']
        df["pricePerRoom"] = df['price'] / (df['bedrooms'] + df['bathrooms'])
        df["bedPerBath"] = df['bedrooms'] / df['bathrooms']
        df["bedBathDiff"] = df['bedrooms'] - df['bathrooms']
        df["bedBathSum"] = df["bedrooms"] + df['bathrooms']
        df["bedsPerc"] = df["bedrooms"] / (df['bedrooms'] + df['bathrooms'])

        df = df.fillna(-1).replace(np.inf, -1)
        return df

    def _descriptionFeatures(self, train, test):
        train['flag'] = 'train'
        test['flag'] = 'test'
        full_data = pd.concat([train, test])
        full_data['description_new'] = full_data.description.apply(lambda x: self._cleanData(x))

        cvect_desc = CountVectorizer(stop_words='english', max_features=200)
        full_sparse = cvect_desc.fit_transform(full_data.description_new)
        # Renaming words to avoid collisions with other feature names in the model
        col_desc = ['desc_' + i for i in cvect_desc.get_feature_names()]
        count_vect_df = pd.DataFrame(full_sparse.todense(), columns=col_desc)
        full_data = pd.concat([full_data.reset_index(), count_vect_df], axis=1)

        train = (full_data[full_data.flag == 'train'])
        test = (full_data[full_data.flag == 'test'])
        return (train,test)

    def _cleanData(self,description):
        stemmer = PorterStemmer()
        regex = re.compile('[^a-zA-Z ]')
        # For user clarity, broken it into three steps
        i = regex.sub(' ', description).lower()
        i = i.split(" ")
        i = [stemmer.stem(l) for l in i]
        i = " ".join([l.strip() for l in i if (len(l) > 2)])  # Keeping words that have length greater than 2
        return i

    def _hcce(self, X_train, X_test):

        X_train = X_train.join(pd.get_dummies(X_train['interest_level'], prefix="pred").astype(int))

        prior_0, prior_1, prior_2 = X_train[["pred_0", "pred_1", "pred_2"]].mean()

        skf = StratifiedKFold(5)
        attributes = product(("building_id", "manager_id"), zip(("pred_1", "pred_2"), (prior_1, prior_2)))
        for variable, (target, prior) in attributes:
            self._hcc_encode(X_train, X_test, variable, target, prior, k=5, r_k=None)
            for train, test in skf.split(np.zeros(len(X_train)), X_train['interest_level']):
                 self._hcc_encode(X_train.iloc[train], X_train.iloc[test], variable, target, prior, k=5, r_k=0.01, update_df=X_train)

        for col in ('building_id', 'manager_id','display_address', 'street_address'):
            X_train, X_test = self._factorize(X_train, X_test, col)

        X_train.drop(["pred_0", "pred_1", "pred_2"],axis=1, inplace=True)

        return X_train,X_test

    def _binarize(self, X_train, X_test):

        fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]  # format features
        X_train["features"] = X_train["features"].apply(fmt)
        X_test["features"] = X_test["features"].apply(fmt)
        features = [f for f_list in list(X_train["features"]) + list(X_test["features"]) for f in f_list]
        ps = pd.Series(features)
        grouped = ps.groupby(ps).agg(len)
        features = grouped[grouped >= 10].index.sort_values().values    # limit to features with >=10 observations
        mlb = MultiLabelBinarizer().fit([features])
        columns = ['feature_' + s for s in mlb.classes_]
        flt = lambda l: [i for i in l if i in mlb.classes_]     # filter out features not present in MultiLabelBinarizer
        X_train = X_train.join(pd.DataFrame(data=mlb.transform(X_train["features"].apply(flt)), columns=columns, index=X_train.index))
        X_test = X_test.join(pd.DataFrame(data=mlb.transform(X_test["features"].apply(flt)), columns=columns, index=X_test.index))

        return X_train, X_test

    def _hcc_encode(self,train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
        """
        See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
        Classification and Prediction Problems" by Daniele Micci-Barreca
        """
        hcc_name = "_".join(["hcc", variable, target])

        grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

        df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
        if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k,
                                        len(test_df))  # Add uniform noise. Not mentioned in original paper

        if update_df is None: update_df = test_df
        if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
        update_df.update(df)
        return

    def _factorize(self,df1, df2, column):
        ps = df1[column].append(df2[column])
        factors = ps.factorize()[0]
        df1[column] = factors[:len(df1)]
        df2[column] = factors[len(df1):]
        return df1, df2

    def addCommonFeature(self,x_train,x_test, feature_name):

        feature_list = list(x_train[feature_name].values) + list(x_test[feature_name].values)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(feature_list)
        x_train[feature_name] = lbl.transform(x_train[feature_name])
        x_test[feature_name] = lbl.transform(x_test[feature_name])