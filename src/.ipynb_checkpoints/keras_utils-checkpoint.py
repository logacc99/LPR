
import numpy as np
import cv2
import time

from os.path import splitext

from src.label                 import Label
from src.utils                 import getWH, nms, crop_region, image_files_from_folder
from src.projection_utils      import getRectPts, find_T_matrix
from src.label 			       import Label, dknet_label_conversion
import darknet.python.darknet  as dn 
from darknet.python.darknet    import detect_vehicle, ocr
import matplotlib.pyplot as plt

def adjust_pts(pts,lroi):
    return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

class DLabel (Label):
    def __init__(self,cl,pts,prob):
        self.pts = pts
        tl = np.amin(pts,1)
        br = np.amax(pts,1)
        Label.__init__(self,cl,tl,br,prob)

def save_model(model,path,verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path,'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print ('Saved to %s' % path)

def load_model(path,custom_objects={},verbose=0):
    from keras.models import model_from_json

# 	path = splitext(path)[0]
# 	with open('%s.json' % path,'r') as json_file:
# 		model_json = json_file.read()
# 	model = model_from_json(model_json, custom_objects=custom_objects)
# 	model.load_weights('%s.h5' % path)
# 	if verbose: print ('Loaded from %s' % path)
# 	return model
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    print("Loading model successfully...")
    return model

def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

def l2_norm(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x-y)*(x-y)))

def reconstruct(Iorig,I,Y,threshold=.5):

    net_stride 	= 2**4
    side 		= ((208. + 40.)/2.)/net_stride # 7.75

    Probs = Y[...,0]
    Affines = Y[...,2:]
    rx,ry = Y.shape[:2]
    ywh = Y.shape[1::-1]
    iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

    xx,yy = np.where(Probs>threshold)

    WH = getWH(I.shape)
    MN = WH/net_stride

    vxx = vyy = 0.5 #alpha

    base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
    labels = []
    labels_frontal = []
    
    for i in range(len(xx)):
        y,x = xx[i],yy[i]
        affine = Affines[y,x]
        prob = Probs[y,x]

        mn = np.array([float(x) + .5,float(y) + .5])

        A = np.reshape(affine,(2,3))
        A[0,0] = max(A[0,0],0.)
        A[1,1] = max(A[1,1],0.)

        pts = np.array(A*base(vxx,vyy)) #*alpha
        pts_MN_center_mn = pts*side
        pts_MN = pts_MN_center_mn + mn.reshape((2,1))

        pts_prop = pts_MN/MN.reshape((2,1))

        labels.append(DLabel(0,pts_prop,prob))
        
        # identity transformation
#         B = np.zeros((2, 3))
#         B[0, 0] = max(A[0, 0], 0)
#         B[1, 1] = max(A[1, 1], 0)
#         pts_frontal = np.array(B*base(vxx, vyy))
#         frontal = normal(pts_frontal, side, mn, MN)
#         labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels,.1)
    # final_labels_frontal = nms(labels_frontal,.1)
    TLps = []
    lps_coor = []
    lps_type = []

    if len(final_labels):
        # print(box)
        # print(final_labels[0].wh()*getWH(Iorig.shape))
        # print(wh_ratio)
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for i,label in enumerate(final_labels):
            box = final_labels[0].pts*getWH(Iorig.shape).reshape((2,1))
            coor = list(zip(box[0], box[1]))
            box_w = l2_norm(coor[0], coor[1])
            box_h = l2_norm(coor[1], coor[2])
            wh_ratio = box_w / box_h
            out_size, lp_type = ((280, 200), 2) if wh_ratio < 2.5 else ((470, 110), 1)

            t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
            ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
            H 		= find_T_matrix(ptsh,t_ptsh)
            Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)
            TLps.append(Ilp)
            lps_coor.append(coor)
            lps_type.append(lp_type)
        
            # coor.append(ptsh)

    return TLps, lps_coor, lps_type

def detect_lp(model,I,max_dim,net_step,threshold):

    min_dim_img = min(I.shape[:2])
    factor 		= float(max_dim)/min_dim_img

    w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
    w += (w%net_step!=0)*(net_step - w%net_step)
    h += (h%net_step!=0)*(net_step - h%net_step)
    Iresized = cv2.resize(I,(w,h))
    T = Iresized.copy()
    T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))
    start 	= time.time()
    Yr 		= model.predict(T)
    Yr 		= np.squeeze(Yr)
    lps_img, lps_coor, lps_type = reconstruct(I,Iresized,Yr,threshold)
    elapsed = time.time() - start
    return lps_img, lps_coor, lps_type, elapsed

def load_yolov2(path_to_data):
    vehicle_weights = f'{path_to_data}/yolo-voc.weights'.encode('utf-8')
    vehicle_netcfg  = f'{path_to_data}/yolo-voc.cfg'.encode('utf-8')
    vehicle_dataset = f'{path_to_data}/voc.data'.encode('utf-8')
    
    vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
    vehicle_meta = dn.load_meta(vehicle_dataset)
    
    print('Load sucessfully...')
    
    return vehicle_net, vehicle_meta

def load_ocr_model(path_to_data):
    ocr_weights = f'{path_to_data}/ocr-net.weights'.encode('utf-8')
    ocr_netcfg  = f'{path_to_data}/ocr-net.cfg'.encode('utf-8')
    ocr_dataset = f'{path_to_data}/ocr-net.data'.encode('utf-8')

    ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = dn.load_meta(ocr_dataset)
    
    print('Load sucessfully...')
    
    return ocr_net, ocr_meta

def vehicle_detection(net, meta, img, vehicle_threshold = 0.5):
    start_time = time.time()
    raw_pred = detect_vehicle(net, meta, img, vehicle_threshold=vehicle_threshold)
    R,_ = raw_pred[0]
    R = [r for r in R if r[0].decode('utf-8') in ['car','bus','motorcycle', 'cycle', 'motorbike']]
    list_vehicles = []
    list_labels = []
    if len(R):
        R = sorted(R, key = lambda x: x[1], reverse = True)
        WH = np.array(img.shape[1::-1],dtype=float)

        for i,r in enumerate(R):
            cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
            tl = np.array([cx - w/2., cy - h/2.])
            br = np.array([cx + w/2., cy + h/2.])
            label = Label(0,tl,br)
            Icar = crop_region(img, label)
            list_vehicles.append(Icar.astype('uint8'))
            xmin, ymin = tuple(label.tl()*WH)
            xmax, ymax = tuple(label.br()*WH)
            list_labels.append((xmin, ymin, xmax, ymax))
    elapsed = time.time() - start_time
    return list_vehicles, list_labels, elapsed

def detect_lp_type(lp_img, lp_type):
    # detect lp_type
    #...
    #...
    
    if lp_type == 1: 
        return lp_img.reshape(1,lp_img.shape[0],lp_img.shape[1], lp_img.shape[2])
    w, h = lp_img.shape[1], lp_img.shape[0]
    center = h//2
    top_part = lp_img[:center, :, :]
    bot_part = lp_img[center:, :, :]
#     fig, ax = plt.subplots(2)
#     ax[0].imshow(top_part)
#     ax[1].imshow(bot_part)
#     plt.show()

    # Return: stacked_img 
    return np.vstack((top_part[None,...], bot_part[None,...]))


def get_ocr_result(ocr_net, ocr_meta, lp_img, lp_type, ocr_threshold=.5, nms_value=.45):
    stacked_img = detect_lp_type(lp_img, lp_type)
    start = time.time()
    raw_pred = ocr(ocr_net, ocr_meta, stacked_img, ocr_threshold = ocr_threshold, nms_value=None)
    lp_str = ''
    for R, (width, height) in raw_pred:
        final_pred = ''
        if len(R):
            L = dknet_label_conversion(R,width,height)
            L = nms(L,.45)
            L.sort(key=lambda x: x.tl()[0])
            final_pred = ''.join([chr(l.cl()) for l in L])
        lp_str += final_pred
    elapsed = time.time()-start
    return lp_str, elapsed