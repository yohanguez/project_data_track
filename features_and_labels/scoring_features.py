import sys
sys.path.insert(0, './features_and_labels')
from features_extraction import Feature_Extractor_Scoring
import glob
import pandas as pd
import time

path_to_portraits_estelle = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/*"
path_to_wedding_pictures = "/Volumes/Untitled/wedding_raw_stills/3/100EOS5D/*.JPG"
path_ID_cluster = "../data/clusters.csv"
path_to_all_wedding_pic_features = "../data/df_wedding_features.csv"
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

def Merge_Wedding_Features_Clusters(path_to_all_wedding_pic_features):
    wedding = pd.read_csv(path_to_all_wedding_pic_features)
    clusters = pd.read_csv(path_ID_cluster, sep=',')
    merge = pd.merge(wedding, clusters, how='inner', left_on='picture_id', right_on='ID')
    merge.to_csv("../data/df_wedding_features_with_cluster.csv")


#tic= time.time()
#Feature_Extraction_wedding(list_wedd_pic ,path_ID_cluster, 'wedding_3')
#tac = time.time()
#print( "For %d pictures, it ran during %0.2f sec"%(len(list_wedd_pic), tac-tic))



#2 : For 979 pictures, it ran during 2382.56 sec
#4 : For 744 pictures, it ran during 2659.34 sec
#6 : For 368 pictures, it ran during 1234.17 sec
#7_1 : For 439 pictures, it ran during 1070.84 sec
#7_2 : For 92 pictures, it ran during 222.62 sec


path_1 = "../data/df_wedding_1_features.csv"
path_2 = "../data/df_wedding_2_features.csv"
path_3 = "../data/df_wedding_4_features.csv"
path_4 = "../data/df_wedding_6_features.csv"
path_5 = "../data/df_wedding_7_1_features.csv"
path_6 = "../data/df_wedding_7_2_features.csv"

df1 = pd.read_csv(path_1)
df2 = pd.read_csv(path_2)
df3 = pd.read_csv(path_3)
df4 = pd.read_csv(path_4)
df5 = pd.read_csv(path_5)
df6 = pd.read_csv(path_6)

df = pd.concat((df1, df2, df3, df4, df5, df6), axis = 0)
df.reset_index(drop=True).to_csv( "../data/df_wedding_features.csv")


Merge_Wedding_Features_Clusters(path_to_all_wedding_pic_features)