from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


#Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt resides")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe Pretrained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", required=True,
                help="Path to the output directory")
args = vars(ap.parse_args())

#Load our serialized model from disk.
print("[INFO] loading model...")
model_file = args["model"]
config_file = args["prototxt"]
output_dir = args["output"]

net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Initialize the video stream and allow the camera sensor to heatup.
print("[INFO] starting the camera...")
# vid = cv2.VideoCapture(0)
vid = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
    #grab the frame and resize it to have a maximume width of 360 pixels.
    frame = vid.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    #get the dimensions of the frame and convert it to blob.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections and
    # Prediction
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y) - coordinates of the bounding box for the
        #object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box of the face along with the associated probability.
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)

    # show the output frame.
    cv2.imshow("Video frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if q is pressed, break the loop.
    if key == ord("c"):
        p = os.path.sep.join([args["output"], "{}.png".format(
            str(total).zfill(5))])

        orig = imutils.resize(orig, width=200)
        cv2.imwrite(p, orig)
        total +=1

        if total > 20:
            print("You have reached the maximume amount of pictures")
            break

    elif key == ord('q') & 0xFF:
        break

# Free your resources.
cv2.destroyAllWindows()
vid.stop()
