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

pathname_portraits_features = "./data/df_portraits_features.csv"
pathname_portaits_labels = "./data/label_portraits.csv"
pathname_wedding_features = "./data/df_wedding_features.csv"
pathname_wedding_labels = "./data/label_wedding.csv"


### define clusters

#>>[paths_clust1, paths_clust2 ..]


### evaluate images on these clusters

#1 - Train/test on portraits
X_df_portrait = pd.read_csv(pathname_portraits_features, sep =',')
y_df_portrait = pd.read_csv(pathname_portaits_labels, sep=';')
merged = merging(X_df_portrait, y_df_portrait)
X_portrait, y_portrait = to_X_y(merged)
test_size = 0.3

for C in [0.1, 1, 10, 100]:
    model = SVR(C = C)
    clf = Scoring(model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(C, clf.evaluate_portraits(y_test))


for C in [0.1, 1, 10, 100]:
    for penalty in ['l1', 'l2']:
        model = LogisticRegression(penalty = penalty, C = C, class_weight = 'balanced')
        clf = Scoring(model)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print(C,penalty, clf.evaluate_portraits(y_test))


for maxdep in [2, 5, 10, 20]:
    model = RandomForestRegressor(max_depth = maxdep)
    clf = Scoring(model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(clf.evaluate_portraits(y_test))


##After grid searching best config =  RandomForestRegressor(max_depth = 2)
