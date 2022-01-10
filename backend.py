# Import lib
import base64
import os, cv2
import datetime
from io import BytesIO
import logging

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
from full_flow import LP_Detector
import src.config as config
from waitress import serve

try: #python3
    from urllib.request import urlopen
except: #python2
    from urllib2 import urlopen



detector = LP_Detector(debug=False)
app = Flask(__name__)
CORS(app)
counter = 0


def api_guilder(request):
    """Check if api is deprecated/have missing keys/send unsupported file"""
    ext_str = str(config.ALLOWED_EXTENSIONS)[1:-1].replace("'","").replace(", ","/")

    status_code = 400 # Bad Request
    if request.files['file'].filename.split('.')[-1].lower() not in config.ALLOWED_EXTENSIONS:
        message = 'Invalid file extension. ' \
                 f'Supported file extensions: {ext_str}'
    # all passed
    else:
        status_code = 200 # OK
        message     = 'Connect and process sucessfully'

    return message, status_code


def process_img_object(bytes_object):
    """convert img file_object to numpy array (BGR)"""
    img = Image.open(BytesIO(bytes_object))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def process_request(request, save_request=False):
    file_object  = request.files['file']
    file_name    = file_object.filename
    file_ext     = file_name.split('.')[-1].lower()
    bytes_object = file_object.read() # data in file_object will be removed right after this

    data     = request.form.to_dict()

    logger.info(f'Request input: {file_name}')
    if save_request:
        time_stamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')
        save_path  = os.path.join(
            config.UPLOAD_DIR,
            "_".join([time_stamp, secure_filename(file_name)]))
        with open(save_path, 'wb') as f:
            f.write(bytes_object)
        logger.info(f'Saved file: {save_path}')

    bgr_img = process_img_object(bytes_object)

    return bgr_img, file_name

@app.route('/recognize', methods=['POST'])
def main_api():
    global counter
    logger.info(f'#{counter} run')
    message, status_code = api_guilder(request)
    resp = {'message': message}
    if status_code == 400:
        logger.info('Invalid API:', message)
        return jsonify(resp), status_code
    else:
        img, file_name = process_request(request,
            save_request      = True)
        try:
            result_dict = detector.full_processing(img, file_name)
        except Exception as e:
            status_code = 500 # Internal Server Error
            # example: ValueError in the OCR core
            resp = {'message': f'{type(e).__name__} in the OCR core'}
            logger.exception(e)
            return jsonify(resp), status_code
        logger.info(f'Results: {result_dict}')
        resp['data'] = result_dict

        counter += 1
        return jsonify(resp), status_code


if __name__ == '__main__':
    # os.environ['FLASK_ENV'] = 'development'
    logger = logging.getLogger('verbose')
    logger.info('API runs')
    app.run(host = "0.0.0.0", port = 1800, debug=False)
    # serve(app, host = "0.0.0.0", port = 1800)