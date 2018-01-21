import glob#
import pandas as pd

#  labeling

raw_pic_list_files = glob.glob('/Volumes/Untitled/wedding_raw_stills/*')
pic_kept_list_files = glob.glob('/Users/estelleaflalo/Desktop/Mariage/wedding_stills/*')

raw_pic_id = []
for folder1 in raw_pic_list_files:
    for folder2 in glob.glob(folder1+'/*'):
        for im in glob.glob(folder2+'/*.JPG'):
            temp = im[::-1][:im[::-1].find('/')][::-1]
            pid = temp[:temp.find(".")]
            raw_pic_id.append(pid)

pic_selected_id = []

for folder1 in pic_kept_list_files:
    for folder2 in glob.glob(folder1+'/*'):
        if folder2[-3:] == 'jpg':
            temp = folder2[::-1][:folder2[::-1].find('/')][::-1]
            pid = temp[:temp.find(".")]
            pic_selected_id.append(pid)
        else:
            for im in glob.glob(folder2+'/*.jpg'):
                temp = im[::-1][:im[::-1].find('/')][::-1]
                pid = temp[:temp.find(".")]
                pic_selected_id.append(pid)

target = pd.DataFrame(index= raw_pic_id, columns = ["Kept"])

for original_pic in raw_pic_id:
    if original_pic in pic_selected_id:
        target[original_pic] =1
    else:
        target[original_pic] = 0


target.to_csv("./data/label_wedding.csv")