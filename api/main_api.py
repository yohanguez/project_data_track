import glob
import sys
sys.path.insert(0, '../features_and_labels')
from features_extraction import Feature_Extractor_Scoring
import pickle
import os
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def profile():
    return(render_template("template.html"))


@app.route('/', methods=['POST'])
def my_form_post():
    rate = request.form['text']
    path_to_images = "/api/static/8*.JPG"
    path_to_no_image = "static/first_pic.JPG"
    main_path = os.path.abspath(os.curdir)
    main_path = main_path[:main_path.rfind("scoring") + len("scoring")]
    model =  main_path + "/models/scoring_model.sav"
    images_cluster = glob.glob(main_path + path_to_images)
    n_kept = int(eval(rate) * len(images_cluster))
    feat_class = Feature_Extractor_Scoring(images_cluster, 1)
    feat = feat_class.Extractor()
    clf = pickle.load(open(model, 'rb'))
    pred = clf.predict(feat)
    indices_pic_kept = np.argsort(pred)[::-1][:n_kept]
    images_kept = np.asarray(images_cluster)[indices_pic_kept]
    result = []
    for im in images_kept:
        result.append(im[im.find('static'):])
    while len(result) < len(images_cluster):
        result.append(path_to_no_image)
    return render_template("index.html", user_image1=result[0], user_image2=result[1], user_image3=result[2], user_image4=result[3], user_image5=result[4], user_image6=result[5], user_image7=result[6], user_image8=result[7])

#@app.route('/')
#@app.route('/index')
#def show_index():
#    full_filename = "static/8O4A8977.JPG"
#    return render_template("index.html", user_image = full_filename)


if __name__=="__main__":
    app.debug = True
    app.run()