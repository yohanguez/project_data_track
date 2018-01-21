import cv2
import imutils
from imutils import face_utils
import dlib
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import math


def people(filename, detector, predictor):
    image = cv2.imread(filename)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    people_faces = []
    people_bodies = []
    center_x = 0
    center_y = 0
    n_peop = 0
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        face = predictor(gray, rect)
        face = face_utils.shape_to_np(face)
        xmin, xmax, ymin, ymax = face[:, 0].min(), face[:, 0].max(), -face[:, 1].max(), -face[:, 1].min()
        width_head = xmax - xmin
        height_head = ymax - ymin
        xminbody = max(xmin - width_head, 0)
        xmaxbody = min(xmin + width_head, image.shape[1]) #0 before
        yminbody = -ymin
        ymaxbody = min(abs(ymin - 8 * height_head), image.shape[0])
        xminhead = max(0, xmin)
        xmaxhead = min(xmax, image.shape[1])
        yminhead = max(-ymax, 0)
        ymaxhead = max(-ymin, 0)
        im_body = image[yminbody: ymaxbody, xminbody: xmaxbody]
        im_head = image[yminhead: ymaxhead, xminhead:xmaxhead]
        im_body = img_to_array(im_body)
        im_head = img_to_array(im_head)
        people_faces.append(im_head)
        people_bodies.append(im_body)
        center_x += (xmaxhead + xminhead) / 2
        center_y += (ymaxhead + yminhead) / 2
        n_peop += 1
    if n_peop > 0 :
        head_center = [center_y / n_peop, center_x / n_peop]
    else :
        head_center =  [0,0]
    return people_faces, people_bodies, n_peop, head_center


def mean_list(l):
    return sum(l) / len(l)


def brightness(im):
    b, g, r = np.mean(im, axis=(0, 1))
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def hsv(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_m, s_m, v_m = np.mean(hsv, axis=(0, 1))
    h_std, s_std, v_std = np.std(hsv, axis=(0, 1))
    return (h_m, s_m, v_m, h_std, s_std, v_std)


def hist_colors(im):
    hist = cv2.calcHist([im], [0], None, [256], [0, 256])
    return hist


def blurry(im):
    temp = im.sum(axis=2).shape
    laplacian = cv2.Laplacian(temp, cv2.CV_64F).var()
    return laplacian


def blurry2(im):
    temp = im.sum(axis=2).shape
    laplacian = np.abs(cv2.Laplacian(temp, cv2.CV_64F)).sum()
    return laplacian


def background(img):
    return 0


def blurry_background(img):
    return 0


def n_people(people_out):
    return people_out[2]


def avg_head_location(people_out):
    return people_out[3]


def blurry_people(people_out, weight_head):
    result = []
    for faces in people_out[0]:
        result.append(blurry(faces) * weight_head)
    for body in people_out[1]:
        try:
            result.append(blurry(body))
        except:
            pass
    return mean_list(result)


def brightness_background(img):
    return 0


def brightness_people(people_out, weight_head):
    result = []
    for faces in people_out[0]:
        result.append(brightness(faces) * weight_head)
    for body in people_out[1]:
        try:
            result.append(brightness(body))
        except:
            pass
    return mean_list(result)


def hsv_background(img):
    return 0


def hsv_people(people_out, weight_head):
    h_m = []
    s_m = []
    v_m = []
    h_std = []
    s_std = []
    v_std = []
    for faces, body in zip(people_out[0], people_out[1]):
        x1, x2, x3, x4, x5, x6 = hsv(faces)
        try:
            y1, y2, y3, y4, y5, y6 = hsv(body)
        except:
            y1, y2, y3, y4, y5, y6 = 0, 0, 0, 0, 0, 0
        z1 = x1 + y1
        z2 = x2 + y2
        z3 = x3 + y3
        z4 = x4 + y4
        z5 = x5 + y5
        z6 = x6 + y6
        h_m.append(z1)
        s_m.append(z2)
        v_m.append(z3)
        h_std.append(z4)
        s_std.append(z5)
        v_std.append(z6)
    return mean_list(h_m), mean_list(s_m), mean_list(v_m), mean_list(h_std), mean_list(s_std), mean_list(v_std)

def feature_matrix(filename, detector, predictor, weight_head):
        img = load_img(filename, detector, predictor)
        im = img_to_array(img)
        people_out = people(filename, detector, predictor)

        result = []
        x0 = avg_head_location(people_out)
        x1 = brightness(im)
        x2, x3, x4, x5, x6, x7 = hsv(im)
        x8 = 0#list(hist_colors(im).reshape(-1))
        x9 = blurry(im)
        if people_out[2] == 0:
            x10 = 0
            x11 = 0
            x12 = 0
            x13 = 0
            x14 = 0
            x15 = 0
            x16 = 0
            x17 = 0
            x18 = 0
        else:
            x10 = n_people(people_out)
            x11 = blurry_people(people_out, weight_head)
            x12 = brightness_people(people_out, weight_head)
            x13, x14, x15, x16, x17, x18 = hsv_people(people_out, weight_head)

        for feat in (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18):
            try:
                result.extend(feat)
            except:
                result.append(feat)
        return result

'''##test
filename0 = "/Users/estelleaflalo/Desktop/target0.JPG"
filename1 = "/Users/estelleaflalo/Desktop/target1.JPG"

image0 = load_img(filename0)
image1 = load_img(filename1)

im0 = img_to_array(image0)
im1 = img_to_array(image1)

peop0 = people(filename0)
peop1= people(filename1)


blurry2(im0)
blurry2(im1)
'''
N_FEATURES = 20

class Feature_Extractor_Scoring():
    def __init__(self, list_files, weight_head):
        self.weight_head = weight_head
        self.detector = dlib.get_frontal_face_detector()
        sel f.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.list_files = list_files

    def Extractor(self):
        features = np.zeros((len(self.list_files), N_FEATURES))
        for i, filename in enumerate(self.list_files):
            features[i, :] = feature_matrix(filename, self.weight_head)
        return features