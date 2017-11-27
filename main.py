# Course:       CS 4732
# Student Name: Thomas Fuller
# Student ID:   000678589
# Assignment #: Test #2, Part #2
# Due Date:     10/30/2017
#
# Signature:    ______________________________________________________________
#               (The signature means that the work is your own, not
#               from somewhere else.)
#
# Score:        ____________________

# ==============================================================================
# FILE: Main.py
# ==============================================================================

from tkinter import filedialog

import cv2
from parts_classifier.classifier import Classifier

classifier = Classifier()

path = filedialog.askopenfilename(title="Select Image...")

# Extract features and classify each identified object
objs, color_img, out, boxes = classifier.extract_features(path)
final_img = classifier.label_objects(color_img, objs)

# Display images
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', color_img)

cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
cv2.imshow('Threshold', out)

cv2.namedWindow('Bounded Boxes', cv2.WINDOW_NORMAL)
cv2.imshow('Bounded Boxes', boxes)

cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)
cv2.imshow('Final Image', final_img)

# Wait for user input
while True:
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()

print("Finished!")
