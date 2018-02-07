import features_VGG16 as ft
import model_clustering as m

path_ori = "/Users/yohanguez/Documents/ITC/ITCCourse/data/data_debugg"
path_dest="/Users/yohanguez/Documents/ITC/ITCCourse/data/data_debugg"

features = ft.features_VGG16(path_ori)
features.compute_features()
features.dump(path_dest)

my_model = m.model_clustering(210, 2)
my_model.pickle_load(path_dest + + '/pic_list.pkl', path_dest + + '/features_list_VGG16.pkl')
my_model.fit_predict()
my_model.print_statistics()