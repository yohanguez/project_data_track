import features_VGG16 as ft
import model_clustering as m

path = "/Users/yohanguez/Documents/ITC/ITCCourse/project_data_track_copy/data/data"

#features = ft.features_VGG16(path)
#features.compute_features()
#features.dump(path)

my_model = m.model_clustering(210, 2)
my_model.pickle_load(path + '/result/pic_list.pkl', path +
                     '/result/features_list_VGG16.pkl')
my_model.fit_predict(path)
my_model.print_statistics()
my_model.create_folder_all(path)


