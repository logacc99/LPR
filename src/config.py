""" PATHS """
INPUT_DIR                       = '/home/giathuan/Programming/TMA/VN_ALPR/dataset/car_long'

OUTPUT_DIR                      = './output/demo'

UPLOAD_DIR                      = './dataset/upload'

""" MODELS """
PATH_TO_VEHICLE_DETECTOR        = './models/yolov5s.pt'

PATH_TO_LP_DETECTOR             = './models/lp-detector/train-9-1-2022/wpod-net_best'

PATH_TO_OCR_MODEL               = './models/ocr/ocr-net'

PATH_TO_TYPE_PREDICTOR          = './models/row_classify_model.h5'

""" THRESHOLDS """
LP_THRESHOLD                    = 0.5

VEHICLE_THRESHOLD               = 0.25

OCR_THRESHOLD                   = 0.4

""" IN-FLOW """
VEHICLE_DETECTION               = True

""" DEBUG """
ACTIVATE_TIMER                  = True

DEBUG                           = False

SAVE_PLOT                       = True

SHOW_PLOT                       = True
