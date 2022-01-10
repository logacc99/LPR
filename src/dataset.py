
import numpy as np
import cv2
import sys
import os
from os.path 					import splitext, basename
from glob import glob
import matplotlib.pyplot as plt
from src.utils import poly_to_bbox, intersect_over_union


def get_true_bbox(img_path, input_dir, img, bbox_pred):
    img_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
    bbox_true, _ = get_label_from_OpenALPR(f'{input_dir}/{img_name}.txt')
    # bbox_true = get_label_from_xml(f'{input_dir}/{img_name}.xml')
    # bbox_true = get_label_from_CCPD(img_name)
    # bbox_true, _ = get_label_from_AOLP(img_path)
    if (len(bbox_true) == 0):
        prob = -1 
    else:
        img = cv2.rectangle(img, tuple(bbox_true[:2]), tuple(bbox_true[2:]), color = (255,0,0), thickness = 2) # red
        img = cv2.rectangle(img, tuple(bbox_pred[:2]), tuple(bbox_pred[2:]), color = (255,255,0), thickness = 2) # yellow
        iou_value = intersect_over_union(bbox_true, bbox_pred)
        prob = iou_value
    return prob, img

def get_label_from_xml(filePath): 
    #print(filename)
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(filePath).getroot()
    except:
        return ()
    coor = []
    boxes = root.findall('object/bndbox')
    for box in boxes:
        for tag in box.findall('*'):
            coor.append(int(tag.text))
        break
    return tuple(coor)

def get_label_from_OpenALPR(filePath):
    import os.path
    if os.path.exists(filePath) == False:
        return ()
    f = open(filePath, 'r')
    line = f.readline().split()
    coor = (int(line[1]), int(line[2]), int(line[1])+int(line[3]), int(line[2])+int(line[4]))
    label = line[5]
    return coor, label

def get_label_from_CCPD(img_name):
    _, box = img_name.split('-')[2:4]
    pts = box.split('_')
    pts = [tuple(map(int, pt.split('&'))) for pt in pts]
    coor = poly_to_bbox(pts)
    return coor

def get_label_from_AOLP(filePath):
    bname = splitext(basename(filePath))[0]
    parent_path = '/'.join(filePath.split('/')[:-2])
    if os.path.exists(filePath) == False:
        return ()

    if os.path.exists(f'{parent_path}/groundtruth_localization/{bname}.txt') == False:
        coor = ''
    else:
        f = open(f'{parent_path}/groundtruth_localization/{bname}.txt', 'r')
        coor = f.readlines()
        coor = [int(float(x[:-1])) for x in coor]
        if coor[0] > coor[2]:
            coor[0],coor[2] = coor[2], coor[0]
        f.close()

    if os.path.exists(f'{parent_path}/groundtruth_recognition/{bname}.txt') == False:
        label = ''
    else:
        f = open(f'{parent_path}/groundtruth_recognition/{bname}.txt', 'r')
        chars = f.readlines()
        chars = [ch[:-1] for ch in chars]
        label = ''.join(chars)
        f.close()

    return coor, label