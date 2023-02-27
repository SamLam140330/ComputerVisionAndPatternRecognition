import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

face_cascade = cv2.CascadeClassifier('./lab5data/haarcascade_frontalface_default.xml')


# Face detection using viola jones
def face_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    roi_gray = None
    if len(faces) == 0:
        return None, None
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
        return roi_gray, faces[0]


# LBP extraction
def lbp_ext(gray_img):
    radius = 2
    n_points = 8
    lbp = local_binary_pattern(gray_img, n_points, radius, 'default')
    n_bins = int(lbp.max() + 1)
    hist, bins = np.histogram(lbp, n_bins, (0, n_bins), density=True)
    return hist


# Training data of LBP feature and label
def train_data():
    feature_list = []
    label_list = []
    train_dir = './lab5data/training/'
    sec_list = os.listdir(train_dir)
    for SecNum in range(len(sec_list)):
        sec_name = sec_list[SecNum]
        sec_dir = os.path.join(train_dir, sec_name)
        img_list = os.listdir(sec_dir)
        for ImgNum in range(len(img_list)):
            image = cv2.imread(os.path.join(train_dir, sec_name, img_list[ImgNum]))
            detected_face, _ = face_detection(image)
            if detected_face is not None:
                lbp_feature = lbp_ext(detected_face)
                feature_list.append(lbp_feature)
                if sec_name == 'fake':
                    label = 0
                elif sec_name == 'real':
                    label = 1
                else:
                    error = "unknown class"
                    raise NotImplementedError(error)
                label_list.append(label)
    # KNN training
    train_feature = np.array(feature_list).astype(np.float32)
    train_label = np.array(label_list)
    return train_feature, train_label


TrainFeature, TrainLabel = train_data()
knn = cv2.ml.KNearest_create()
knn.train(TrainFeature, cv2.ml.ROW_SAMPLE, TrainLabel)

# KNN Testing
test_way = 'img'
if test_way == 'img':
    TestFeature, TestLabel = train_data()

    ret, results, neighbours, dist = knn.findNearest(TestFeature, k=5)
    matches = results.squeeze() == TestLabel
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / len(results)
    print('The accuracy of face presentation attack detection: {:.2f} %'.format(accuracy))
elif test_way == 'cam':
    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        DetectedFace, rect = face_detection(frame)
        if DetectedFace is not None:
            cv2.imshow('DetectedFace', DetectedFace)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            LBPFeature = lbp_ext(DetectedFace)
            LBPFeature = np.array([LBPFeature]).astype(np.float32)
            ret, results, neighbours, dist = knn.findNearest(LBPFeature, k=5)
            if results.squeeze() == 1:
                label_text = 'real face'
            else:
                label_text = 'fake face'
            draw_text(frame, label_text, rect[0], rect[1] - 5)
            cv2.imshow('result', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('No face detected, please try again')
            break
    video_capture.release()
    cv2.destroyAllWindows()
