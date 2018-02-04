import sys
sys.path.insert(0, './features_and_labels')
from features_extraction import Feature_Extractor_Scoring
import glob
import pandas as pd
import time

path_to_portraits_estelle = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/*"
path_to_wedding_pictures = "/Volumes/Untitled/wedding_raw_stills/6/*.JPG"
path_ID_cluster = "../data/clusters.csv"
list_portraits = glob.glob(path_to_portraits_estelle)
list_wedd_pic = glob.glob(path_to_wedding_pictures)

def Feature_Extraction_Portraits(list_portraits):
    feat_class = Feature_Extractor_Scoring(list_portraits, 1)
    feat = feat_class.Extractor()
    feat_class.save_to_df('portraits')

def Feature_Extraction_wedding(list_wedd_pic ,path_ID_cluster, title):
    feat_class = Feature_Extractor_Scoring(list_wedd_pic, 1)
    feat = feat_class.Extractor()
    wedding = feat_class.save_to_df(title)
    clusters = pd.read_csv(path_ID_cluster, sep=',')
    merge = pd.merge(wedding, clusters, how='inner', left_on='picture_id', right_on='ID')
    merge.to_csv("../data/df_wedding_features_with_cluster.csv")

    #2hours

tic= time.time()
Feature_Extraction_wedding(list_wedd_pic ,path_ID_cluster, 'wedding_6')
tac = time.time()
print( "For %d pictures, it ran during %0.2f sec"%(len(list_wedd_pic), tac-tic))


#2 : For 979 pictures, it ran during 2382.56 sec
#4 : For 744 pictures, it ran during 2659.34 sec
#6 : For 368 pictures, it ran during 1234.17 sec
