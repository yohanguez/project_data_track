import sys
sys.path.insert(0, './models')
from model_scoring import Scoring
sys.path.insert(0, './features_and_labels')
from features_extraction import to_X_y, merging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

pathname_portraits_features = "./data/df_portraits_features.csv"
pathname_portaits_labels = "./data/label_portraits.csv"
pathname_wedding_features = "./data/df_wedding_features.csv"
pathname_wedding_labels = "./data/label_wedding.csv"
pathname_wedding_features_clusters = "./data/df_wedding_features_with_cluster.csv"

test_size = 0.3
PERCENTAGE_KEPT_PER_CLUSTER = 0.1

X_df_portrait = pd.read_csv(pathname_portraits_features, sep =',')
y_df_portrait = pd.read_csv(pathname_portaits_labels, sep=';')
merged = merging(X_df_portrait, y_df_portrait)
X_portrait, y_portrait = to_X_y(merged)

model = RandomForestRegressor(max_depth=2)
clf = Scoring(model)
clf.fit(X_portrait, y_portrait)


X_df_wedding = pd.read_csv(pathname_wedding_features_clusters)
id = X_df_wedding["picture_id"]
y_df_wedding = pd.read_csv(pathname_wedding_labels)
merged = merging(X_df_wedding, y_df_wedding)
for cluster in np.unique(merged["cluster"]):
    merged_temp = merged[merged["cluster"]==cluster]
    X_wedding_tmp, y_wedding_tmp = to_X_y(merged_temp)
    pred_from_portrait = clf.predict(X_wedding_tmp)


