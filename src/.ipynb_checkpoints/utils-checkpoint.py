
import numpy as np
import cv2
import sys
import os
from os.path 					import splitext, basename

from glob import glob
import matplotlib.pyplot as plt


def im2single(I):
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.

def draw_plate_box(img, coor, thickness=2): 
#     pts=[]  
#     x_coordinates=cor[0][0]
#     y_coordinates=cor[0][1]
#     # store the top-left, top-right, bottom-left, bottom-right 
#     # of the plate license respectively
#     for i in range(4):
#         pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
#     pts = np.array(pts, np.int32)
#     pts = pts.reshape((-1,1,2))
    pts = []
    for x,y in coor:
        pts.append((int(x), int(y)))
    
    pts_reshaped = np.array(pts, np.int32)
    pts_reshaped = pts_reshaped.reshape((-1,1,2))
    cv2.polylines(img,[pts_reshaped],True,(0,255,0),thickness)
    # print(pts)
    bbox = poly_to_bbox(pts)
    img = cv2.rectangle(img, bbox[:2], bbox[2:], color = (255,255,0), thickness = 2) #yellow
    return img, bbox

def get_original_box(vehicle_box, plate_box):
    ans_box = list()
    x_offset = int(vehicle_box[0])
    y_offset = int(vehicle_box[1])
    ans_box.append(plate_box[0]+x_offset)
    ans_box.append(plate_box[1]+y_offset)
    
    ans_box.append(plate_box[2]+x_offset)
    ans_box.append(plate_box[3]+y_offset)
    return ans_box

def poly_to_bbox(pts):
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    retval = cv2.boundingRect(pts)
    rec = (retval[0], retval[1], retval[0]+retval[2], retval[1]+retval[3])
    return rec

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

def get_true_bbox(img_path, input_dir, img, bbox_pred):
    img_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
    bbox_true, _ = get_label_from_OpenALPR(f'{input_dir}/{img_name}.txt')
    # bbox_true = get_label_from_xml(f'{input_dir}/{img_name}.xml')
    # bbox_true = get_label_from_CCPD(img_name)
    # bbox_true, _ = get_label_from_AOLP(img_path)
    if (len(bbox_true) == 0):
        prob = -1 
    else:
        img = cv2.rectangle(img, tuple(bbox_true[:2]), tuple(bbox_true[2:]), color = (255,0,0), thickness = 2) #red
        img = cv2.rectangle(img, tuple(bbox_pred[:2]), tuple(bbox_pred[2:]), color = (255,255,0), thickness = 2) #yellow
        iou_value = iou(bbox_true, bbox_pred)
        prob = iou_value
    return prob, img

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_value = interArea / float(boxAArea) #+ boxBArea - interArea)
    # return the intersection over union value
    return iou_value


def getWH(shape):
	return np.array(shape[1::-1]).astype(float)


def IOU(tl1,br1,tl2,br2):
	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())
	
	intersection_wh = np.maximum(np.minimum(br1,br2) - np.maximum(tl1,tl2),0.)
	intersection_area = np.prod(intersection_wh)
	area1,area2 = (np.prod(wh1),np.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area


def IOU_labels(l1,l2):
	return IOU(l1.tl(),l1.br(),l2.tl(),l2.br())


def IOU_centre_and_dims(cc1,wh1,cc2,wh2):
	return IOU(cc1-wh1/2.,cc1+wh1/2.,cc2-wh2/2.,cc2+wh2/2.)


def nms(Labels,iou_threshold=.5):

	SelectedLabels = []
	Labels.sort(key=lambda l: l.prob(),reverse=True)
	
	for label in Labels:

		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels(label,sel_label) > iou_threshold:
				non_overlap = False
				break

		if non_overlap:
			SelectedLabels.append(label)

	return SelectedLabels


def image_files_from_folder(folder,upper=True):
	extensions = ['jpg','jpeg','png']
	img_files  = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder,ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder,ext.upper()))
	return img_files


def is_inside(ltest,lref):
	return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()


def crop_region(I,label,bg=0.5):

	wh = np.array(I.shape[1::-1])

	ch = I.shape[2] if len(I.shape) == 3 else 1
	tl = np.floor(label.tl()*wh).astype(int)
	br = np.ceil (label.br()*wh).astype(int)
	outwh = br-tl

	if np.prod(outwh) == 0.:
		return None

	outsize = (outwh[1],outwh[0],ch) if ch > 1 else (outwh[1],outwh[0])
	if (np.array(outsize) < 0).any():
		pause()
	Iout  = np.zeros(outsize,dtype=I.dtype) + bg

	offset 	= np.minimum(tl,0)*(-1)
	tl 		= np.maximum(tl,0)
	br 		= np.minimum(br,wh)
	wh 		= br - tl

	Iout[offset[1]:(offset[1] + wh[1]),offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1],tl[0]:br[0]]

	return Iout

def hsv_transform(I,hsv_modifier):
	I = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
	I = I + hsv_modifier
	return cv2.cvtColor(I,cv2.COLOR_HSV2BGR)

def IOU(tl1,br1,tl2,br2):
	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())
	
	intersection_wh = np.maximum(np.minimum(br1,br2) - np.maximum(tl1,tl2),0.)
	intersection_area = np.prod(intersection_wh)
	area1,area2 = (np.prod(wh1),np.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area

def IOU_centre_and_dims(cc1,wh1,cc2,wh2):
	return IOU(cc1-wh1/2.,cc1+wh1/2.,cc2-wh2/2.,cc2+wh2/2.)


def show(I,wname='Display'):
	cv2.imshow(wname, I)
	cv2.moveWindow(wname,0,0)
	key = cv2.waitKey(0) & 0xEFFFFF
	cv2.destroyWindow(wname)
	if key == 27:
		sys.exit()
	else:
		return key

def multiplot(imgs, titles = [], fig_size = (12, 8)):
    n = len(imgs)
    fig, ax = plt.subplots(1, n, figsize = fig_size) 
    for i in range(n):
        ax[i].imshow(imgs[i])
        if len(titles):
            ax[i].set_title(titles[i])
    plt.show()
    plt.close(fig)