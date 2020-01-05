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
import os, sys
import pickle
import pandas
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image
import glob

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
            img_csv = pandas.read_csv(ann_file, sep=',', header = None, skiprows=1, chunksize=1, dtype=str)
            label_csv = pandas.read_csv(lable_map,sep=',', header = None, chunksize=1)
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_dir + ann)
        for row in label_csv:
            label_map[str(row[0].iloc[0])] = str(row[1].iloc[0])
        
        for row in img_csv:
            iid = str(row[0].iloc[0])
            if not iid in imgs:
                imgs[iid] = {}
                imgs[iid]['object'] = []
                imgs[iid]['filename'] = os.path.join(img_dir, iid + '.jpg')
                try:
                    im = Image.open(imgs[iid]['filename'])
                    imgs[iid]['width'], imgs[iid]['height'] = im.size
                except:
                    npath = glob.glob(os.path.join(img_dir, iid) + '.*')
                    imgs[iid]['filename'] = npath[0]
                    im = Image.open(imgs[iid]['filename'])
                    imgs[iid]['width'], imgs[iid]['height'] = im.size
            
            label_id = str(row[2].iloc[0])
            label_name = label_map[label_id]
            if label_name in seen_labels:
                seen_labels[label_name] += 1
            else:
                seen_labels[label_name] = 1
            if len(labels) > 0 and label_name not in labels:
                continue
            else:
                obj = {
                    'name': label_name,
                    'xmin': int(round(float(row[4].iloc[0]) * imgs[iid]['width'])),
                    'ymin': int(round(float(row[5].iloc[0]) * imgs[iid]['height'])),
                    'xmax': int(round(float(row[6].iloc[0]) * imgs[iid]['width'])),
                    'ymax': int(round(float(row[7].iloc[0]) * imgs[iid]['height']))}
                imgs[iid]['object'].append(obj)
        print(imgs)
        for key, img in imgs.items():
            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels
