import glob
import sys
sys.path.insert(0, '../features_and_labels')
from features_extraction import Feature_Extractor_Scoring
import pickle
import os
import numpy as np

def im_id_kept(rate, path_to_images):
    main_path = os.path.abspath(os.curdir)
    main_path = main_path[:main_path.rfind("scoring") + len("scoring")]
    images_cluster = glob.glob(main_path + path_to_images )
    model =  main_path + "/models/scoring_model.sav"
    clf = pickle.load(open(model, 'r'))
    feat_class = Feature_Extractor_Scoring(images_cluster, 1)
    feat = feat_class.Extractor()
    pred = clf.predict(feat)
    n_kept = min(int(rate * len(images_cluster)) , len(images_cluster))
    indices_pic_kept = np.argsort(pred)[::-1][:n_kept]
    return np.asarray(images_cluster)[indices_pic_kept]


print(im_id_kept(0.5, "/api/static/8*.JPG"))