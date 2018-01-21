from features_extraction import Feature_Extractor_Scoring



list_file =["/Users/estelleaflalo/Desktop/target0.JPG",  "/Users/estelleaflalo/Desktop/target1.JPG"]
feat_class = Feature_Extractor_Scoring(list_file, 1)
feat = feat_class.Extractor()
feat_class.save_to_df()