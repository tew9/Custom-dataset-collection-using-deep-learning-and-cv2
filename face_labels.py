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
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream and allow the camera sensor to heatup.
print("[INFO] starting the camera...")
