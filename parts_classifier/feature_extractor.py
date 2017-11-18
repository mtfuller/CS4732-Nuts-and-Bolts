import cv2
import numpy as np

KERNEL_5x5 = np.ones((5,5),np.uint8)
KERNEL_9x9 = np.ones((9,9),np.uint8)

MIN_AREA = 0.001
MAX_AREA = 0.9

class FeatureExtractor:
    def __init__(self):
        self._current_id = 0
        self.rows = []
    def collect_feature_data(self, filename, **kwargs):
        output_dir = kwargs['output'] if 'output' in kwargs else None

        original = cv2.imread(filename,1)
        parts, bounded_boxes = calculate_bounding_boxes(original)

        for part in parts:
            img = get_subimage(original, part)
            features = extract_features(img)
            if output_dir:
                filename = output_dir+str(self._current_id)+'.png'
                cv2.imwrite(filename, img)
            else:
                filename = ''
            self.rows.append([self._current_id,img.shape[1],img.shape[0],*features,filename,'NONE'])
            print("Found Number of Objects",len(self.rows))
            self._current_id += 1

def extract_features(img):
    area_ratio, area_img = extract_area_ratio(img)
    box_scale, area_img = extract_box_scale(img)
    total_corners, corner_img = extract_total_corners(img)
    error, area_img = extract_circularity_error(img)
    perimeter_norm, area_img = extract_perimeter_norm(img)
    return [area_ratio, box_scale, total_corners, error, perimeter_norm]

def calculate_bounding_boxes(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Morphilogical Gradient
    gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, KERNEL_5x5)

    # Closing Image
    closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, KERNEL_9x9)

    # Thresholding
    ret,thresh1 = cv2.threshold(closing,35,255,cv2.THRESH_BINARY)

    # Calculating Bounding Boxes
    rects, contours, h = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA = 0.005
    MAX_AREA = 0.9

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

    return parts, found_parts_img

def get_subimage(img, bounding_box):
    x,y,w,h = bounding_box
    return img[y:y+h, x:x+w]

def extract_area_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    pixels_object = np.count_nonzero(thr)
    pixels_box = thr.shape[0]*thr.shape[1]
    return (pixels_object / pixels_box), thr

def extract_box_scale(img):
    height = float(img.shape[0])
    width = float(img.shape[1])
    if width <= height:
        return width/float(height), img
    else:
        return height/float(width), img

def extract_total_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thr = np.float32(thr)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    dst = cv2.cornerHarris(closing, 2, 3, 0.05)
    number_of_corners = np.count_nonzero(dst[dst>0.03*dst.max()])
    dst = cv2.dilate(dst,None)
    corner_img = img.copy()
    corner_img[dst>0.03*dst.max()]=[0,0,255]
    return number_of_corners, corner_img

def extract_circularity_error(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    b, object_conts, h = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_areas = [cv2.contourArea(object_conts[i]) for i in range(len(object_conts))]
    max_contour_idx = np.argmax(contour_areas)
    area_object = contour_areas[max_contour_idx]

    radius = img.shape[0]/2.0 if img.shape[0] > img.shape[1] else img.shape[1]/2.0
    area_circle = np.pi * radius**2

    circularity_error = abs(area_object - area_circle) / ((2*radius)**2)

    shape_img = img.copy()
    cv2.drawContours(shape_img,[object_conts[max_contour_idx]],-1,(0,255,0),3)
    return circularity_error, shape_img

def extract_perimeter_norm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    b, object_conts, h = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_perimeters = [cv2.arcLength(object_conts[i],True) for i in range(len(object_conts))]
    max_contour_idx = np.argmax(contour_perimeters)

    perimeter_object = contour_perimeters[max_contour_idx]
    perimeter_norm = perimeter_object / (2.0*img.shape[0] + 2.0*img.shape[1])

    shape_img = img.copy()
    cv2.drawContours(shape_img,[object_conts[max_contour_idx]],-1,(0,255,0),3)
    return perimeter_norm, shape_img
