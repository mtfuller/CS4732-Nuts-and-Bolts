import csv, os
import glob
from parts_classifier.feature_extractor import FeatureExtractor

SOURCE_DIR = "./source_img/"
OUTPUT_DIR = './data/'
OUTPUT_IMAGE_DIR = OUTPUT_DIR + 'img/'

images = glob.glob(SOURCE_DIR+"*.JPG")

for DIR in [OUTPUT_DIR,OUTPUT_IMAGE_DIR]:
    if not os.path.exists(DIR):
        os.makedirs(DIR)

ext = FeatureExtractor()

for img in images:
    ext.collect_feature_data(img, output=OUTPUT_IMAGE_DIR)

with open(OUTPUT_DIR+'data.csv','w') as f:
    writer=csv.writer(f)
    writer.writerow(['id','x','y','width','height','area_ratio','box_scale','total_corners','error','perimeter','file','label'])
    for row in ext.rows:
        writer.writerow(row)

print("FINISHED!!!")
