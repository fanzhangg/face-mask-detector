from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


def mask_image(img, min_confidence=0.5):


    # Load model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load the face mask detector
    print("[INFO] loading face mask detector model")
    model = load_model("mask_detector.model")

    # Load and copy the image
    image = cv2.imread(img)
    orig = image.copy()
    h, w = image.shape[:2]

    # Construct a blob from the Image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    print("[INFO] computing face detection...")
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detection
    for i in range(0, detections.shape[2]):
        # Extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > min_confidence:
            # Compute the x,y coordinates of the bouding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = (min(w-1, endX), min(h-1, endY))

            # Extract the face ROI
            face = image[startY: endY, startX: endX]
            # Convert it from BGR to RGB channel ordering
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Resize
            face = cv2.resize(face, (224, 224))
            # Preprocess
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Pass the face through the model to determine if the face has a mask
            (mask, withoutMask) = model.predict(face)[0]

            # Determine the class label and color for the boundary
            if mask > withoutMask:
                label = "Yes"
                color = (143, 157, 42)
            else:
                label = "No"
                color = (70, 57, 230)

            # Add the probability
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display the label and bounding box
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            blk = np.zeros(image.shape, np.uint8)
            cv2.rectangle(blk, (startX, startY), (endX, endY), color, 2)
            image = cv2.addWeighted(image, 1.0, blk, max(mask, withoutMask), 1)
            # image = cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    return image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())
    image = mask_image(args["image"])
    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
