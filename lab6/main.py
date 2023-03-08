import os

import cv2
import numpy as np


def detect_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./lab6data/haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    if len(face) == 0:
        return None, None
    (x, y, w, h) = face[0]
    return gray_img[y:y + w, x:x + h], face[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    all_faces = []
    all_labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            face = cv2.resize(face, (200, 300))
            if face is not None:
                all_faces.append(face)
                all_labels.append(label)
    return all_faces, all_labels


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    face = cv2.resize(face, (200, 300))
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img


print("Preparing data...")
subjects = ["", "Ramiz Raja", "Elvis Presley"]
faces, labels = prepare_training_data("./lab6data/training-data")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
print("Data prepared")

print("Predicting images...")
test_img1 = cv2.imread("./lab6data/test-data/test1.jpg")
test_img2 = cv2.imread("./lab6data/test-data/test2.jpg")
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

print("Showing both images")
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
