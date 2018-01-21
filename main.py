import sys
sys.path.insert(0, './models')
from model_scoring import Scoring
sys.path.insert(0, './features_and_labels')
from features_extraction import to_X_y

from sklearn.svm import SVR

pathname_portraits_features = "./data.df_portraits_features.csv"
pathname_portaits_labels = "./label_portraits.csv"
pathname_wedding_features = "./data.df_wedding_features.csv"
pathname_wedding_labels = "./label_wedding.csv"


### define clusters

#>>[paths_clust1, paths_clust2 ..]


### evaluate images on these clusters

#1 - Train/test on portraits

X, y = to_X_y(pathname_portraits_features, pathname_portaits_labels)
model = SVR(C = 10)
clf = Scoring(X, y, model)
X_train, X_test, y_train, y_test = clf.split()
clf.fit_predict()
print(clf.evaluate())
