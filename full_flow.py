import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from    vehicle    import Vehicle_Detector
from    src.model_utils         import load_model, get_lp_detection, load_yolov2, \
                                load_ocr_model, get_ocr_result
import  src.config  as config
from    src.utils  import poly_to_bbox, get_reference_box, multiplot
import  cv2
import  tensorflow  as tf
import  numpy       as np
import  matplotlib.pyplot       as plt
from    tqdm        import tqdm
import time

tf.__version__
print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))


class LP_Detector():
    def __init__(self, debug=True):
        """ Model """
        print('Loading WPOD-NET...')
        self.wpod_net = \
            load_model(config.PATH_TO_LP_DETECTOR)

        print('Loading Vehicle Detector...')
        self.vehicle_detector = Vehicle_Detector(
            weights     =config.PATH_TO_VEHICLE_DETECTOR,
            threshold   =config.VEHICLE_THRESHOLD)

        print('Loading OCR Model...')
        self.ocr_net, self.ocr_meta = \
            load_yolov2(config.PATH_TO_OCR_MODEL)

        print('Loading Type Predictor...')
        self.type_predictor = tf.keras.models.load_model(config.PATH_TO_TYPE_PREDICTOR)

        """ Paths """
        self.output_dir = config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        """ Supporting variables """
        self.vehicle_flag = config.VEHICLE_DETECTION
        self.save_plot = config.SAVE_PLOT
        self.show_plot = config.SHOW_PLOT
        self.debug = config.DEBUG
        if self.debug:
            os.makedirs('debug/', exist_ok=True)

        print('Initialize done.')

    def detect_plate(self, vehicle_img):
        lps_img, lps_coor, lps_type, elapsed = get_lp_detection(
            self.wpod_net,
            vehicle_img,
            config.LP_THRESHOLD,
            self.type_predictor)

        Ilp = None
        lp_type = None
        lp_coor = None
        detected = False

        if len(lps_img):
            detected = True
            Ilp = lps_img[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            Ilp = Ilp * 255.
            Ilp = Ilp.astype(np.uint8)
            lp_coor = lps_coor[0]
            lp_type = lps_type[0]

        return detected, Ilp, lp_coor, lp_type

    def contrast_increase(self, img):
        lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final

    def ocr(self, plate_img, plate_type):
        lp_str, ocr_time = get_ocr_result(
            self.ocr_net,
            self.ocr_meta,
            plate_img,
            plate_type,
            ocr_threshold=config.OCR_THRESHOLD)
        return lp_str

    def detect_plate_ocr(self, org_img, vehicle_boxes):
        ocr_results = []
        plate_boxes = []
        lp_warped = []
        """ DETECT PLATES """
        for idx in range(len(vehicle_boxes)):
            vbox = vehicle_boxes[idx]
            vehicle_img = org_img[vbox[1]:vbox[3], vbox[0]:vbox[2]]
            detected, lp_img, lp_coor, lp_type = \
                self.detect_plate(vehicle_img)

            plate_boxes.append(poly_to_bbox(lp_coor))
            lp_warped.append(lp_img)

            """ APPLY OCR """
            if detected:
                lp_img = self.contrast_increase(lp_img)
                lp_str = self.ocr(lp_img, lp_type)
                ocr_results.append(lp_str + f'({lp_type})')
            else:
                ocr_results.append("None")
        return ocr_results, plate_boxes, lp_warped

    def box_label(self, image, vehicle_boxes, plate_boxes, list_ocr, label_vehicle):
        for index in range(0, len(vehicle_boxes)):
            """ DRAW VEHICLE BOX """
            vehicle_box = vehicle_boxes[index]
            label = label_vehicle[index] + " - " + list_ocr[index] # CAR/TRUCK/BUS/ MOTOR + CAR PLATE
            lw = max(round(sum(image.shape) / 2 * 0.003), 2)
            p1, p2 = (int(vehicle_box[0]), int(vehicle_box[1])), (int(vehicle_box[2]), int(vehicle_box[3]))
            cv2.rectangle(image, p1, p2, (0, 248, 0), thickness=lw, lineType=cv2.LINE_AA)

            """ DRAW PLATE BOX """
            plate_box = plate_boxes[index]
            if plate_box:
                plate_box = get_reference_box(vehicle_box, plate_box)
                cv2.rectangle(
                    image, 
                    (int(plate_box[0]), int(plate_box[1])), 
                    (int(plate_box[2]), int(plate_box[3])), 
                    (248, 248, 0), thickness=lw, lineType=cv2.LINE_AA)
            
            """ TEXT SETTINGS """
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            
            """ LABEL FITS OUTSIDE BOX """
            outside = p1[1] - h - 3 >= 0  
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, (128, 128, 128), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, (255, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)

        """ DRAW COUNTING LINE """
        # start_point = (0, round(image.shape[0] * 0.6))
        # end_point = (image.shape[1], round(image.shape[0] * 0.6))
        # cv2.line(image, start_point, end_point, (0, 0, 248), 9)
        return image

    def full_processing(self, org_img, detect_vehicle = True, filename = None):
        """ STEP 1: DETECT VEHICLES IN IMAGE """
        if detect_vehicle:
            vehicle_boxes, label_vehicle = self.vehicle_detector.detect(image=org_img)
        else:
            vehicle_boxes = [[0, 0, org_img.shape[1], org_img.shape[0]]]
            label_vehicle = ['']

        """ STEP 2: DETECT PLATE ON VEHICLES AND APPLY OCR """
        plate_ocr_results, plate_boxes, lp_warped = self.detect_plate_ocr(org_img, vehicle_boxes)

        """ STEP 3: VISUALIZE RESULT """
        image_draw = self.box_label(org_img.copy(), vehicle_boxes, plate_boxes, plate_ocr_results, label_vehicle)

        """ DEBUG """
        if self.debug:
            if filename is None:
                filename = str(int(time.time()*1e6))
            save_dir = f'debug/{filename}'
            os.makedirs(save_dir, exist_ok=True)
            for idx, box in enumerate(vehicle_boxes):
                vehicle_img = org_img[
                                int(box[1]):int(box[3]), 
                                int(box[0]):int(box[2])]
                lp_box = plate_boxes[idx]
                if lp_box:
                    lp_img = lp_warped[idx]
                    cv2.rectangle(vehicle_img, (int(lp_box[0]),int(lp_box[1])), (int(lp_box[2]),int(lp_box[3])), (255, 255, 0), 2)
                    multiplot(
                            [vehicle_img, lp_img], 
                            titles = [label_vehicle[idx], plate_ocr_results[idx]],
                            show_plot = True,
                            save_dir = os.path.join(save_dir, f'{idx}.jpg'))
                else:
                    plt.show(vehicle_img)
                    plt.title(label_vehicle[idx]+'- None')
                    plt.show()

        return image_draw, vehicle_boxes, plate_ocr_results 

# def demo_video(video_file):
#     detector = LP_Detector()
#     cap = cv2.VideoCapture(video_file)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = cap.read()
#     skip_frame = 4
#     index_frame = 0
#     list_images = []
#     while ret:
#         ret, frame = cap.read()
#         if index_frame % skip_frame != 0:
#             index_frame += 1
#             continue
#         list_images.append(detector.full_processing(frame, detector_vehicle)[0])
#         index_frame += 1

#     # STEP 4
#     print("STEP 4: generating video...")
#     generate_video(list_images)


def detect_batch(image_paths, lp_detector,save_dir = None):
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fname = image_path.split('/')[-1].split('.')[0]

        # Read vehicle boxes
        with open(f'/content/drive/MyDrive/UFPR-ALPR-dataset/labels/detection/testing/{fname}.txt', 'r') as f:
            vehicle_labels = f.read()
            xmin, ymin, w, h = list(map(int, vehicle_labels.split(' ')[1:]))

        processed_img, vehicle_boxes, ocr_results = lp_detector.full_processing(image[ymin:ymin+h, xmin:xmin+w, :], False, fname) 
        
        if save_dir:
            os.makedirs(save_dir, exist_ok = True)
            
            with open(os.path.join(save_dir, f'{fname}.txt'), 'w') as f:
                f.write(ocr_results[0])
            
            # Save result
            cv2.imwrite(os.path.join(save_dir, f'{fname}.png'), processed_img) 

if __name__ == '__main__':
    demo_video('/content/drive/MyDrive/FinalProject/data/videoplayback.mp4')
