import cv2
import numpy as np

#camera = cv2.VideoCapture(0)
widthHeight = 160
classFile = 'face.names'
classNames = []
confThreshold = 0.3
nmsThreshold = 0.01
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

modelConfiguration = '/Users/sujithsaisripadam/Desktop/temp/yolov3-face.cfg'
modelWeights = '/Users/sujithsaisripadam/Desktop/temp/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    height, width, channel = img.shape
    Bounding_box = []
    classIds = [] 
    confs = [] 

    for output in outputs:
        for detection in output:
            scores = detection[5:]  
            classId = np.argmax(scores)  
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2) 
                Bounding_box.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(Bounding_box, confs, confThreshold, nmsThreshold)
    coordinates = []
    a = []
    b = []
    for i in indices:
        i = [i]
        i = i[0]
        box = Bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        coordinates.append([x,y,w,h])
        a.append(classNames[classIds[i]].upper())
        b.append(int(confs[i]*100))
    return coordinates,a,b

def object_detect(img):
    img = cv2.imread(img)
    #img = img
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = []
    for i in net.getUnconnectedOutLayers():
        i = [i]
        outputNames.append(layersNames[i[0] - 1])
    outputs = net.forward(outputNames)
    return findObjects(outputs, img)
"""
cap = cv2.VideoCapture(0)
counter = 0
while True:
    print(counter)
    counter+=1
    success, img = cap.read()
    img = cv2.resize(img,(1080,720))
    cv2.imwrite("input_image.jpg",img)
    cv2.imshow("Real Image",img)

    coordinates,a,b = object_detect("input_image.jpg")
    for i,j,k in zip(coordinates,a,b):
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img, f'{j}{k}%',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imshow('Detected',img)
    cv2.waitKey(1)
"""

"""----------------------------------video---------------------------------"""

import cv2

def read_and_display_video(video_path):
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture('/Users/sujithsaisripadam/Desktop/temp/video1.mp4')
    counter = 0
    while True:
        print(counter)
        counter+=1
        success, img = cap.read()
        img = cv2.resize(img,(720,480))
        cv2.imwrite("input_image.jpg",img)
        cv2.imshow("Realtime video",img)

        coordinates,a,b = object_detect("input_image.jpg")
        for i,j,k in zip(coordinates,a,b):
            x = i[0]
            y = i[1]
            w = i[2]
            h = i[3]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 1)
            cv2.putText(img, f'{j}{k}%',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imshow('Detected',img)
        cv2.waitKey(1)


if __name__ == "__main__":
    video_path = "/Users/sujithsaisripadam/Desktop/temp/video1.mp4"  # Replace with the path to your video file
    read_and_display_video(video_path)






