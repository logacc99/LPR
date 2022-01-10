
import 	numpy as np
import 	cv2
import 	sys
import 	os
from 	os.path 				import splitext, basename
from 	glob                   	import glob
import 	matplotlib.pyplot    	as plt
import 	functools
import 	time

def read_image(img_path):
    name = splitext(basename(img_path))[0]
    image = cv2.imread(img_path)
    return image, name

def im2single(I):
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.	

def draw_plate_box(img, coor, thickness=2): 
    pts = []
    for x,y in coor:
        pts.append((int(x), int(y)))
    
    pts_reshaped = np.array(pts, np.int32)
    pts_reshaped = pts_reshaped.reshape((-1,1,2))
    cv2.polylines(img,[pts_reshaped],True,(0,255,0),thickness)
    bbox = poly_to_bbox(pts)
    img = cv2.rectangle(img, bbox[:2], bbox[2:], color = (255,255,0), thickness = 2) #yellow
    return img, bbox

def get_reference_box(big_box, small_box):
    output = list()
    x_offset = int(big_box[0])
    y_offset = int(big_box[1])
    output.append(small_box[0]+x_offset)
    output.append(small_box[1]+y_offset)
    
    output.append(small_box[2]+x_offset)
    output.append(small_box[3]+y_offset)
    return output

def poly_to_bbox(pts):
    if pts is None:
        return None
        
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    retval = cv2.boundingRect(pts)
    rec = (retval[0], retval[1], retval[0]+retval[2], retval[1]+retval[3])
    return rec

def intersect_over_union(boxA, boxB):
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
	assert ((wh1>=.0).all() and (wh2>=.0).all())

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
    GroupLabels = []
    Labels.sort(key=lambda l: l.prob(),reverse=True)

    for label in Labels:
        non_overlap = True
        for idx, sel_label in enumerate(SelectedLabels):
            if IOU_labels(label,sel_label) > iou_threshold:
                non_overlap = False

                # Check whether new character in group
                group_label = GroupLabels[idx]
                cur_char = label.cl()
                list_chars = [lb.cl() for lb in group_label]
                if cur_char not in list_chars:
                    GroupLabels[idx].append(label)

                break

        if non_overlap:
            SelectedLabels.append(label)
            GroupLabels.append([label])

    return SelectedLabels, GroupLabels


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

def show(I,wname='Display'):
	cv2.imshow(wname, I)
	cv2.moveWindow(wname,0,0)
	key = cv2.waitKey(0) & 0xEFFFFF
	cv2.destroyWindow(wname)
	if key == 27:
		sys.exit()
	else:
		return key

def timing(func=None, *, activate=True, return_time=False, factor=1000,
           split_before=False, split_after=False, times=1):
	"""Print out the running time of the function, support repeated run
	Will do nothing if activate = False
	Result is in mili-second.

	example 1:
	>>> @timing(activate=True)
	>>> def test():
	>>>     sleep(0.1)
	>>>     return 'yay'
	>>> test()
	Runtime of test                           100.38185 ms
	'yay'

	example 2:
	>>> @timing(activate=True)
	>>> def crop(): ...
	>>>
	>>> @timing(activate=True)
	>>> def remove_text(): ...
	>>>
	>>> @timing(activate=True, split_after=True)
	>>> def full_flow(): ... # include crop() and remove_text()
	>>>
	>>> full_flow()
	Runtime of crop                           146.19160 ms
	Runtime of remove_text                   3316.30707 ms
	Runtime of full_flow                     4403.89156 ms
	######################### 0 ##########################
	Runtime of crop                            72.94130 ms
	Runtime of remove_text                   1034.61838 ms
	Runtime of full_flow                     1443.43138 ms
	######################### 1 ##########################
	"""
	def decor_timing(func):
		time_run = 0
		@functools.wraps(func)
		def wrap_func(*args,**kwargs):
			nonlocal time_run

			time1 = time.time()
			for _ in range(times):
				ret = func(*args,**kwargs)
			time2 = time.time()
			run_time = time2 - time1

			if split_before:
				print(f"{' '+str(time_run)+' ':#^56}")
				time_run +=1

			ti = '' if times==1 else f"x{times} "
			print('Runtime of {}{:<30s}{:>12.2f} ms'.format(ti,
				func.__name__, run_time*factor))

			if split_after:
				print(f"{' '+str(time_run)+' ':#^54}")
				time_run +=1

			if return_time:
				return run_time*factor
			else:
				return ret
		if activate:
			return wrap_func
		else:
			return func

	if func is None:
		return decor_timing
	else:
		return decor_timing(func)

def multiplot(imgs, titles = [], fig_size = (12, 8), show_plot = False, save_dir = None):
	n = len(imgs)
	fig, ax = plt.subplots(1, n, figsize = fig_size) 
	for i in range(n):
		ax[i].imshow(imgs[i])
		if len(titles):
			ax[i].set_title(titles[i])

	if show_plot:
		plt.show()

	if save_dir is not None:
		plt.savefig(save_dir)
		
	plt.close(fig)

def generate_video(list_images, video_name = 'example.mp4'):
    size = (list_images[0].shape[1], list_images[0].shape[0])
    vid_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    for i in range(len(list_images)):
        vid_writer.write(list_images[i])
    print("DONE!")
