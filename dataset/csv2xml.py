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
import pickle
import pandas
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image

def parse_tfrecord_annotation(ann_file, img_dir, lablefile, cache_name, labels=[]):
    '''
    ann_file: /data/annotations/train-annotations-bbox.csv
    img_dir: /data/images/train/
    '''
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}

        try:
            csv = pandas.read_csv(ann_file).values
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_file)
            continue
        img = {}
        for row in csv:
            if not row[0] in img:
                fn = row[0] + '.jpg'
                img[row[0]] = {
                    'filename': fn,
                    'path': img_dir + fn
                }
                im = Image.open(img[row[0]]['path'])
                img[row[0]]['size'] = {}
                img[row[0]]['size']['width'], img[row[0]]['size']['height'] = im.size
                img[row[0]]['size']['depth'] = 3
            img[row[0]]['object'] = []
            img[row[0]]['object'].append({
                'name': row[2]
            })


        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels