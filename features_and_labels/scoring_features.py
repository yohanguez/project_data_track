import sys
sys.path.insert(0, './features_and_labels')
from features_extraction import Feature_Extractor_Scoring
import glob

path_to_portraits_estelle = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/*"
path_to_wedding_pictures = ""

#Features extraction for portraits
list_portraits = glob.glob(path_to_portraits_estelle)
feat_class = Feature_Extractor_Scoring(list_portraits, 1)
feat = feat_class.Extractor()
feat_class.save_to_df('portraits')


#Features extraction for wedding pictures

#list_wedd_pic = glob.glob(path_to_wedding_pictures)
#feat_class = Feature_Extractor_Scoring(list_wedd_pic, 1)
#feat = feat_class.Extractor()
#feat_class.save_to_df('wedding')
#