'''
### CSV Format ###
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
000026e7ee790996,freeform,/m/07j7r,1,0.071905,0.145346,0.206591,0.391306,0,1,1,0,0
### File Format ###
/data/
  - /images/
    - /train/
    - /validation/
    - /test/
  - /annotations/
    - train-annotations-bbox.csv
    - validation-annotations-bbox.csv
    - test-annotations-bbox.csv
    - class-descriptions-boxable.csv  
'''
import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import os
import pickle
import pandas
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image

def parse_openimg_annotation(ann_file, img_dir, lable_map, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        imgs = {}
        label_map = {}
        
        try:
            img_csv = pandas.read_csv(ann_file).values
            label_csv = pandas.read_csv(lable_map).values
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue

        for row in label_csv:
            label_map[row[0]] = row[1]
        
        for row in img_csv:
            if not row[0] in imgs:
                imgs[row[0]] = {}
                imgs[row[0]]['filename'] = os.path.join(img_dir, row[0] + '.jpg')
                im = Image.open(imgs[row[0]]['filename'])
                imgs[row[0]]['width'], imgs[row[0]]['height'] = im.size
                imgs[row[0]]['object'] = []
            
            label_id = row[2]
            label_name = label_map[label_id]
            if label_name in seen_labels:
                seen_labels[label_name] += 1
            else:
                seen_labels[label_name] = 1
            if len(labels) > 0 and label_name not in labels:
                break
            else:
                obj = {
                    'name': label_name,
                    'xmin': int(round(float(row[4] * img['width']))),
                    'ymin': int(round(float(row[5] * img['height']))),
                    'xmax': int(round(float(row[6] * img['width']))),
                    'ymax': int(round(float(row[7] * img['height'])))}
                imgs[row[0]]['object'].append(obj)

        for key, img in imgs:
            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels