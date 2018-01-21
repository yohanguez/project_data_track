from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#pathname for the dataframe with 273 features for 1000 pictures
pathname_feature = "/Users/estelleaflalo/Desktop/ITC/DataSciencesTrack/Project/df_1000.csv"
pathname_label = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/Final_Machine_Learning/portrait/dataframes/output_train.csv"


class Scoring():
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def split(self, test_size = 0.33):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.pred = self.model.predict(self.X_test)
        return self.pred

    def evaluate(self):
        return mean_squared_error(self.test, self.pred)




wedding_pictures = glob.glob("/Users/estelleaflalo/Desktop/ITC/DataSciencesTrack/Project/huppa1/*.jpg")


pic_id_test = []
for filename in wedding_pictures:
    temp = filename[::-1][:filename[::-1].find('/')][::-1]
    pid = temp[:temp.find(".")]
    pic_id_test.append(pid)



WEIGHT_HEAD = 1
N_FEATURES = 273
TEST_SIZE = len(pic_id_test)
features2 = np.zeros((TEST_SIZE, N_FEATURES))
for i, filename in enumerate(wedding_pictures):
    features2[i, :] = feature_matrix(filename, WEIGHT_HEAD)
    if i%100==0:
        print i

features2 = preprocessing.scale(features2)
pred_wedding = model.predict(features2)

filename0 = "/Users/estelleaflalo/Desktop/target0.JPG"
filename1 = "/Users/estelleaflalo/Desktop/target1.JPG"
In [765]:

X0 = feature_matrix(filename0, 1)
X0 = preprocessing.scale(X0)
​
X1 = feature_matrix(filename1, 1)
X1 = preprocessing.scale(X1)

model.predict(X1)
model.predict(X0)


best_pictures_sorted = [wedding_pictures[i] for i in np.argsort(pred_wedding)[::-1]]



