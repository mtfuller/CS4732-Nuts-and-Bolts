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
# FILE: classifier.py
# ==============================================================================

import cv2
import numpy as np
import pickle

MODEL_PATH = './models/nut_bolt_test.sav'
CLASSES = ['NUT', 'BOLT', 'WASHER']

MIN_AREA = 0.001
MAX_AREA = 0.9

class Classifier:
    def __init__(self):
        self.load_model(MODEL_PATH)
        self._current_id = 0
        self.rows = []

    def extract_features(self, filename):
        """Extracts features from a number of objects that are identified from
        the given image.
        """
        color_img = cv2.imread(filename,1)
        original = cv2.imread(filename,1)
        parts, out, bounded_boxes = calculate_bounding_boxes(original)

        objects = []

        # Extract features from each part identified.
        for part in parts:
            bin_img = get_subimage(out, part)
            features = extract_features(bin_img)
            objects.append({
                'id': self._current_id,
                'x': part[0],
                'y': part[1],
                'w': part[2],
                'h': part[3],
                'features': features,
                'label': 'UNKNOWN'
            })
            self._current_id += 1

        return objects, color_img, out, bounded_boxes

    def collect_feature_data(self, filename, **kwargs):
        """Returns a list of all metadata and features of each object identified
        in the image.
        """
        output_dir = kwargs['output'] if 'output' in kwargs else None

        original = cv2.imread(filename,1)
        parts, out, bounded_boxes = calculate_bounding_boxes(original)

        for part in parts:
            color_img = get_subimage(original, part)
            bin_img = get_subimage(out, part)
            features = extract_features(bin_img)
            if output_dir:
                filename = output_dir+str(self._current_id)+'.png'
                cv2.imwrite(filename, color_img)
            else:
                filename = ''
            self.rows.append([self._current_id,part[0],part[1],bin_img.shape[1],bin_img.shape[0],*features,filename,'NONE'])
            print("Found Number of Objects",len(self.rows))
            self._current_id += 1

    def load_model(self, filename):
        """Load the classification model.
        """
        self.model = pickle.load(open(filename, 'rb'))

    def classify_object(self, obj):
        """Classifies a single object based on the features given.
        """
        features = np.array(obj['features'])
        features = np.append(features, (features[2] / (obj['w'] * obj['h'])))
        cls = self.model.predict([features])
        return CLASSES[cls[0]-1]

    def label_objects(self, img, objects):
        """Returns an image that contains rendered text and bounding boxes that
        identifies and classifies each object.
        """
        final = img.copy()
        for obj in objects:
            x, y, w, h = obj['x'],obj['y'],obj['w'],obj['h']
            final = cv2.rectangle(final,(x,y),(x+w,y+h),(0,255,0),10)
            cls = self.classify_object(obj)
            cv2.putText(final, cls, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
        return final

def extract_features(img):
    """Main feature extraction function
    """
    area_ratio, area_img = extract_area_ratio(img)
    box_scale, area_img = extract_box_scale(img)
    total_corners, corner_img = extract_total_corners(img)
    error, area_img = extract_circularity_error(img)
    perimeter_norm, area_img = extract_perimeter_norm(img)
    return [area_ratio, box_scale, total_corners, error, perimeter_norm]

def calculate_bounding_boxes(img):
    """Returns a list of the sizes and coordinates of the bounding boxes that
    surround identified objects.
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    med_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, small_kernel)

    ret2,out = cv2.threshold(gradient,50,255,cv2.THRESH_BINARY)

    out = cv2.dilate(out,med_kernel,iterations=1)

    rects, contours, h = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    parts = []
    found_parts_img = img.copy()

    num_of_objects = 0
    for i in range(len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        area_ratio = (w*h) / (img.shape[0] * img.shape[1])
        if (area_ratio >= MIN_AREA and area_ratio <= MAX_AREA):
            num_of_objects += 1
            parts.append(tuple([x,y,w,h]))
            found_parts_img = cv2.rectangle(found_parts_img,(x,y),(x+w,y+h),(0,255,0),10)

    return parts, out, found_parts_img

def get_subimage(img, bounding_box):
    """Returns a sub-image of the given bounding box from the image.
    """
    x,y,w,h = bounding_box
    return img[y:y+h, x:x+w]

def extract_area_ratio(img):
    """Returns the area ratio from the sub-image. (Percent of white pixels).
    """
    pixels_object = np.count_nonzero(img)
    pixels_box = img.shape[0]*img.shape[1]
    return (pixels_object / pixels_box), img

def extract_box_scale(img):
    """Returns a normalized box scale for the sub-image.
    """
    height = float(img.shape[0])
    width = float(img.shape[1])
    if width <= height:
        return width/float(height), img
    else:
        return height/float(width), img

def extract_total_corners(img):
    """Applies Harris corner detection and returns how many corners exist in
    the subimage.
    """
    dst = cv2.cornerHarris(img, 2, 3, 0.05)
    number_of_corners = np.count_nonzero(dst[dst>0.03*dst.max()])
    return number_of_corners, img

def extract_circularity_error(img):
    """Returns the error in area between the object's calculated area and
    the ideal area of a circle.
    """
    b, object_conts, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_areas = [cv2.contourArea(object_conts[i]) for i in range(len(object_conts))]
    max_contour_idx = np.argmax(contour_areas)
    area_object = contour_areas[max_contour_idx]

    radius = img.shape[0]/2.0 if img.shape[0] > img.shape[1] else img.shape[1]/2.0
    area_circle = np.pi * radius**2

    circularity_error = abs(area_object - area_circle) / ((2*radius)**2)
    return circularity_error, img

def extract_perimeter_norm(img):
    """Returns a value for normalized perimeter around and object.
    """
    b, object_conts, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_perimeters = [cv2.arcLength(object_conts[i],True) for i in range(len(object_conts))]
    max_contour_idx = np.argmax(contour_perimeters)

    perimeter_object = contour_perimeters[max_contour_idx]
    perimeter_norm = perimeter_object / (2.0*img.shape[0] + 2.0*img.shape[1])
    return perimeter_norm, img
