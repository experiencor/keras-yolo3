import os
import cv2
import pickle
import numpy as np


def parse_coco_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(ann_dir)):
            img = {'object':[]}

            all_lines = []

            ann_path = os.path.join(ann_dir,ann)
            with open(ann_path,'rt') as label_file:
                raw = label_file.read()
                all_lines = [l.strip() for l in raw.split('\n') if l.strip()]

            image_filename = ann.replace('.txt','.jpg')
            image_filepath = os.path.join(img_dir,image_filename)
            cvimage = cv2.imread(image_filepath)
            if cvimage is None:
                print("Skipping .... {}".format(image_filename))
                continue
            h,w,_ = cvimage.shape

            img['filename'] = image_filepath
            img['width'] = w
            img['height'] = h
                
            for l in all_lines:
                class_index,xcen,ycen,width,height = l.split(' ')                
                obj = {}

                xmin = max(float(xcen) - float(width) / 2, 0)
                xmax = min(float(xcen) + float(width) / 2, 1)
                ymin = max(float(ycen) - float(height) / 2, 0)
                ymax = min(float(ycen) + float(height) / 2, 1)

                xmin = int(w * xmin)
                xmax = int(w * xmax)
                ymin = int(h * ymin)
                ymax = int(h * ymax)

                obj['xmin'] = xmin
                obj['ymin'] = ymin
                obj['xmax'] = xmax
                obj['ymax'] = ymax
                obj['name'] = class_index

                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1

                img['object'] += [obj]

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels


def test():
    import json
    import random
    
    try:
        os.remove("test.pkl")
    except:
        pass

    config_path = "config.json"

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, _ = parse_coco_annotation(  config['train']['train_annot_folder'], 
                                            config['train']['train_image_folder'], 
                                            "test.pkl", 
                                            config['model']['labels'])

    annot_info = random.choice(train_ints)

    colors = {  '0':(255,0,0),
                '1':(0,255,0),
                '2':(0,0,255),}

    cvimage = cv2.imread(annot_info['filename'])

    for bbox in annot_info['object']:
        x1 = bbox['xmin']
        y1 = bbox['ymin']
        x2 = bbox['xmax']
        y2 = bbox['ymax']
        color = colors[bbox['name']]
        cvimage = cv2.rectangle(cvimage,(x1,y1),(x2,y2),color,thickness=1)
    cv2.imshow("display",cvimage)
    cv2.waitKey(0)


 
if __name__ == '__main__':
    test()
