from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def loadNetworks():
    """
    Load the nets for the face detection and mask detection from the disc
    :return: the nets for the face detection and mask detection
    """
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model("mask_detector.model")
    return faceNet, maskNet


def predictMask(image, faceNet, maskNet):
    """
    Detect all faces, and predict each face's probability of wearing mask
    :param image: The input image
    :param faceNet: The network for face detection
    :param maskNet: The network for mask detection
    :return: The locations of each face, and the predicted probability of wearing a mask
    """
    # Construct a blob of a frame
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out the weak detection
        if confidence > 0.5:
            # Compute the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Make sure the bounding box inside the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = image[startY:endY, startX:endX]  # Extract the face ROI

            if face.size == 0:
                print("Skip the face")
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert it from BGR to RGB channel
            face = cv2.resize(face, (224, 224))  # Resize
            face = img_to_array(face)  # Pre-process
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)

    return locations, predictions


def maskImage(image, locs, preds):
    for box, pred in zip(locs, preds):
        startX, startY, endX, endY = box
        mask, withoutMask = pred

        if mask > withoutMask:
            label = "Yes"
            color = (143, 157, 42)
        else:
            label = "No"
            color = (70, 57, 230)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return image


def detectOnVideo():
    """
    Detect if people wear masks on the video stream
    :return:
    """
    print("* Start video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)     # Warm up the video

    faceNet, maskNet = loadNetworks()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        locs, preds = predictMask(frame, faceNet, maskNet)

        frame = maskImage(frame, locs, preds)
        cv2.imshow('Mask Detection', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    detectOnVideo()
