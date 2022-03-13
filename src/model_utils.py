
import numpy as np
import cv2
import time
import base64
import os

from    src.label                   import Label
from    src.utils                   import getWH, nms, crop_region, \
                                    image_files_from_folder, poly_to_bbox, \
                                    timing, im2single
from    src.projection_utils        import getRectPts, find_T_matrix
from    src.label 			        import Label, dknet_label_conversion
import  darknet.python.darknet      as dn
import  matplotlib.pyplot           as plt
import  tensorflow                  as tf
from    src.config                  import ACTIVATE_TIMER

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
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    return model

def load_yolov2(path_to_data):
    model_name = os.path.basename(path_to_data)
    weights = f'{path_to_data}.weights'.encode('utf-8')
    netcfg  = f'{path_to_data}.cfg'.encode('utf-8')
    data = f'{path_to_data}.data'.encode('utf-8')
    
    net  = dn.load_net(netcfg, weights, 0)
    meta = dn.load_meta(data)
    
    return net, meta

def load_ocr_model(path_to_data):
    ocr_weights = f'{path_to_data}/ocr-net.weights'.encode('utf-8')
    ocr_netcfg  = f'{path_to_data}/ocr-net.cfg'.encode('utf-8')
    ocr_dataset = f'{path_to_data}/ocr-net.data'.encode('utf-8')

    ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = dn.load_meta(ocr_dataset)

    return ocr_net, ocr_meta

def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

def l2_norm(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x-y)*(x-y)))

# def get_vehicle_detection(net, meta, img, vehicle_threshold = 0.5):
#     start_time = time.time()
#     raw_pred = dn.do_vehicle_detection(net, meta, img, vehicle_threshold=vehicle_threshold)
#     R,_ = raw_pred[0]
#     R = [r for r in R if r[0].decode('utf-8') in ['car','bus', 'motorbike']]
#     list_vehicles = []
#     list_labels = []
#     if len(R):
#         R = sorted(R, key = lambda x: x[1], reverse = True)
#         WH = np.array(img.shape[1::-1],dtype=float)

#         for i,r in enumerate(R):
#             cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
#             tl = np.array([cx - w/2., cy - h/2.])
#             br = np.array([cx + w/2., cy + h/2.])
#             label = Label(0,tl,br)
#             Icar = crop_region(img, label)
#             list_vehicles.append(Icar.astype('uint8'))
#             xmin, ymin = tuple(label.tl()*WH)
#             xmax, ymax = tuple(label.br()*WH)
#             list_labels.append((xmin, ymin, xmax, ymax))
#     elapsed = time.time() - start_time
#     return list_vehicles, list_labels, elapsed

def reconstruct(Iorig,I,Y,type_predictor,threshold=.5):

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


    final_labels, _ = nms(labels, 0.1)
    TLps = []
    lps_coor = []
    lps_type = []

    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for i,label in enumerate(final_labels):
            pts_normalized = label.pts*getWH(Iorig.shape).reshape((2,1))
            coor = list(zip(pts_normalized[0], pts_normalized[1]))
            rect = [max(0, c) for c in poly_to_bbox(coor)]
            plate_img = (Iorig[rect[1]:rect[3], rect[0]:rect[2], :]*255).astype(np.uint8)
            
            # Detect license plate is 1 row or 2 rows
            lp_type = detect_lp_type(coor, threshold = 0.5)
            # lp_type = 1
            out_size = (160, 120) if lp_type == 2 else (240, 80)
            
            t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
            ptsh 	= np.concatenate((pts_normalized, np.ones((1,4))))
            H 		= find_T_matrix(ptsh,t_ptsh)
            Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)
            TLps.append(Ilp)
            lps_coor.append(coor)
            lps_type.append(lp_type)
            # coor.append(ptsh)
            break

    return TLps, lps_coor, lps_type

def detect_lp_type(list_points, threshold = 0.6):
    p1, p2, p3, p4 = list_points
    d1 = np.linalg.norm(np.array(p1)-np.array(p2))
    d2 = np.linalg.norm(np.array(p2)-np.array(p3))
    ratio = min(d1, d2) / max(d1, d2)
    if ratio > threshold:
        return 2
    else:
        return 1


# def detect_lp_type(lp_img, type_predictor):
#     resized = cv2.resize(lp_img, (224,224), interpolation = cv2.INTER_AREA)
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     img = tf.expand_dims(gray, 0)

#     predictions = type_predictor.predict(img)
#     score = tf.nn.softmax(predictions[0])
#     lpType_names = [1, 2]
#     return lpType_names[np.argmax(score)]

def preprocess_by_type(lp_img, lp_type):
    if lp_type == 1: 
        return lp_img.reshape(1,lp_img.shape[0],lp_img.shape[1], lp_img.shape[2])
    w, h = lp_img.shape[1], lp_img.shape[0]
    center = h//2
    margin = int(h*0.1)
    top_part = lp_img[:center+margin, :, :]
    bot_part = lp_img[center-margin:, :, :]

    # Return: stacked_img 
    return np.vstack((top_part[None,...], bot_part[None,...]))

def get_lp_detection(model, img, threshold, type_predictor):
    Dmin = 288.
    Dmax = 608.
    net_step = 2**4
    ratio = float(max(img.shape[:2]))/min(img.shape[:2])
    side  = int(ratio*Dmin)
    bound_dim = min(side + (side%(net_step)),Dmax)
    I = im2single(img)
    min_dim_img = min(I.shape[:2])
    factor 		= float(bound_dim)/min_dim_img

    w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
    w += (w%net_step!=0)*(net_step - w%net_step)
    h += (h%net_step!=0)*(net_step - h%net_step)
    Iresized = cv2.resize(I,(w,h))
    T = Iresized.copy()
    T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))
    start 	= time.time()
    Yr 		= model.predict(T)
    Yr 		= np.squeeze(Yr)
    lps_img, lps_coor, lps_type = reconstruct(I,Iresized,Yr,type_predictor,threshold)
    elapsed = time.time() - start
    
    return lps_img, lps_coor, lps_type, elapsed

def post_process(label_grouped, region = None):
    default_output = [group[0] for group in label_grouped]
    if region is None:
        return default_output
    
    if region == 'brazil':
        output = []
        fix_pairs = [
            {'letter':'O', 'digit':'0'}, 
            {'letter':'I', 'digit':'1'},
            {'letter':'S', 'digit':'5'},
            {'letter':'B', 'digit':'8'}
            ]
        if len(label_grouped) == 7:
            for idx, group_chars in enumerate(label_grouped):
                letters = [char for char in group_chars if not chr(char.cl()).isdigit()]
                digits = [char for char in group_chars if chr(char.cl()).isdigit()]
                selected = group_chars[0]
                if idx < 3:
                    # Get letter only
                    if len(letters):
                        selected = letters[0]

                    # Fix character
                    for fix_pair in fix_pairs:
                        if chr(selected.cl()) == fix_pair['digit']:
                            selected.set_class(ord(fix_pair['letter']))
                else:
                    # Get digit only
                    if len(digits):
                        selected = digits[0]

                    # Fix character
                    for fix_pair in fix_pairs:
                        if chr(selected.cl()) == fix_pair['letter']:
                            selected.set_class(ord(fix_pair['digit']))

                output.append(selected)
        else:
            output = default_output
    
    return output

def get_ocr_result(ocr_net, ocr_meta, lp_img, lp_type, ocr_threshold=.5, nms_value=.45):
    stacked_img = preprocess_by_type(lp_img, lp_type)
    start = time.time()
    raw_pred = dn.do_ocr(ocr_net, ocr_meta, stacked_img, ocr_threshold = ocr_threshold, nms_value=None)
    lp_str = ''
    final_L = []
    for i, (R, (width, height)) in enumerate(raw_pred):
        if len(R):
            L = dknet_label_conversion(R,width,height)

            """ Debug """
            # for l in L:
            #     tl = l.tl()
            #     br = l.br()
            #     h, w = lp_img.shape[:2]
            #     print(chr(l.cl()), l.prob())
            #     cv2.rectangle(stacked_img[i], (int(tl[0]*width), int(tl[1]*height)), 
            #                 (int(br[0]*width), int(br[1]*height)), (255,0,0), 2)
            # from google.colab.patches import cv2_imshow
            # cv2_imshow(stacked_img[i])
            """ End - Debug """
            L, L_grouped = nms(L,.35)
            
            L.sort(key=lambda x: x.tl()[0])
            L_grouped.sort(key=lambda x: x[0].tl()[0])

            # for group in L_grouped:
            #     cls = [chr(lb.cl()) for lb in group]
            #     print(cls)

            final_L.extend(L_grouped)
    
    L_processed = post_process(final_L, region=None)
    lp_str = ''.join([chr(l.cl()) for l in L_processed])
    # print(lp_str)
    elapsed = time.time()-start
    return lp_str, elapsed