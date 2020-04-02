import os
import imutils
from imutils import face_utils
from imutils.face_utils import rect_to_bb
import cv2
import numpy as np
import dlib
from PIL import Image
import math
import scipy.misc


FACE_DETECTOR = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
EYE_DETECTOR = cv2.CascadeClassifier("haarcascade_eye.xml")


PATH = "AVR_data/"
OUTPUT_path = "processed_images/"

DIRS = os.listdir(PATH)
IMG_PATHS = []

for dir in DIRS:
    for img in os.listdir(PATH + dir):
        path = {'dir': dir, 'img': img}
        IMG_PATHS.append(path)


def generate_path_string(PATH, path):
    return PATH + path['dir'] + '/' + path['img']

def read_image(path):
    img = cv2.imread(generate_path_string(PATH, path), 0)
    img = np.asarray(img)
    return img

def save_image(img, path):
    directory = OUTPUT_path + path['dir']
    if not os.path.exists(directory):
        os.makedirs(directory)
    out_path = generate_path_string(OUTPUT_path, path)
    cv2.imwrite(out_path, img)

def distance(p, q):
    return math.sqrt((q[0] - p[0])**2  + (q[1] - p[1])**2)

def detect_face(img, path):
    faces = FACE_DETECTOR.detectMultiScale(img, 1.3, 5)

    try:
        face_x, face_y, face_w, face_h = faces[0]
        offset = 10
        img = img[int(face_y-offset):int(face_y+face_h+offset), int(face_x-offset):int(face_x+face_w+offset)]
        return img
    except IndexError:
        return img


def detect_eyes(img, path):
    eyes = EYE_DETECTOR.detectMultiScale(img)
    index = 0

    if len(eyes) < 2:
        print("Bad Image, can't detect eyes: {}".format(path['img']))
        return
    if len(eyes) > 2:
        return


    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
          eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
          eye_2 = (eye_x, eye_y, eye_w, eye_h)
        index += 1

    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    return {'left': {'x': left_eye_x, 'y': left_eye_y, 'center': left_eye_center}, 'right': {'x': right_eye_x, 'y': right_eye_y, 'center': right_eye_center}}

def rotate_img(img, eyes, path):
    if eyes['left']['y'] < eyes['right']['y']:
        point_3rd = (eyes['right']['x'], eyes['left']['y'])
        direction = 1 
    else:
        point_3rd = (eyes['left']['x'], eyes['right']['y'])
        direction = -1 

    a = distance(eyes['left']['center'], point_3rd)
    b = distance(eyes['right']['center'], eyes['left']['center'])
    c = distance(eyes['right']['center'], point_3rd)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)
    angle = (angle * 180) / math.pi
    if direction == -1:
       angle = 90 - angle
    img = Image.fromarray(img)
    img = np.array(img.rotate(direction * angle))
    return img

def scale_img(img, eyes):
    dy = eyes['right']['center'][1] - eyes['left']['center'][1]
    dx = eyes['right']['center'][0] - eyes['left']['center'][0]
    dist = np.sqrt((dx**2) + (dy**2))
    scale = 128/dist

    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return img

def draw_oval(img,rect):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    shape = predictor(img, rect)
    shape = face_utils.shape_to_np(shape)
    index = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    detected_points = shape[index[0]:index[1]]
    jaw_tip = detected_points[8]
    jaw_left = detected_points[0]
    jaw_right = detected_points[len(detected_points)-1]
    centre = (
                int((jaw_left[0] + jaw_right[0])/2),
                int((jaw_left[1] + jaw_right[1])/2)
            )
    minor_axis = distance(jaw_tip, centre)
    major_axis = distance(jaw_left, jaw_right)/2
    mask = np.zeros_like(img)
    mask = cv2.ellipse(mask,centre, (major_axis, minor_axis), 0, 0, 0, (0,0,0), -1)
    result = np.bitwise_and(img,mask)
    return result


for img_path in IMG_PATHS:

    img = read_image(img_path)

    img = detect_face(img, img_path)

    eyes = detect_eyes(img, img_path)

    if eyes:
        img = rotate_img(img, eyes, img_path)

        img = scale_img(img, eyes)

    detector = dlib.get_frontal_face_detector()
    rects = detector(img, 2)
    if rects:
        rect = rects[0]
        outline = draw_oval(img, rect)
        (x, y, w, h) = rect_to_bb(rect)
        face = draw_oval(outline,rect)
        try:
            img = imutils.resize(face[y:y + h, x:x + w], width=256)
        except ZeroDivisionError:
            img = img


    img = imutils.resize(img, width=256)
    save_image(img,img_path)

