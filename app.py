# import the necessary packages
import os
import sys
import requests
import ssl
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file


from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import convertToJPG

from os import path
import torch

import fastai
from deoldify.visualize import *
from pathlib import Path
import traceback


torch.backends.cudnn.benchmark=True


os.environ['CUDA_VISIBLE_DEVICES']='0'

app = Flask(__name__)


# define a predict function as an endpoint
@app.route("/process_video", methods=["POST"])
def process_video():

    input_path = generate_random_filename(upload_directory,"jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    try:
        url = request.json["source_url"]
        render_factor = int(request.json["render_factor"])
    except (TypeError):
        traceback.print_exc()
        return {'message': 'Not Acceptable'}, 406 

    try:
        video_path = video_colorizer.colorize_from_url(source_url=url, file_name=input_path, render_factor=render_factor)
        callback = send_file(output_path, mimetype='application/octet-stream')

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        pass
        clean_all([
            input_path,
            output_path
            ])


# define a predict function as an endpoint
@app.route("/process_image", methods=["POST"])
def process_image():

    input_path = generate_random_filename(upload_directory,"jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))
 
    try:
        url = request.json["source_url"]
        render_factor = int(request.json["render_factor"])
    except (TypeError):
        traceback.print_exc()
        return {'message': 'input error'}, 406
        
    try:
        download(url, input_path)

        try:
            image_colorizer.plot_transformed_image(path=input_path, figsize=(20,20),
                render_factor=render_factor, display_render_factor=True, compare=False)
        except:
            convertToJPG(input_path)
            image_colorizer.plot_transformed_image(path=input_path, figsize=(20,20),
            render_factor=render_factor, display_render_factor=True, compare=False)

        callback = send_file(output_path, mimetype='image/jpeg')
        
        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        pass
        clean_all([
            input_path,
            output_path
            ])


if __name__ == '__main__':
    global upload_directory
    global results_img_directory
    global image_colorizer, video_colorizer

    upload_directory = '/data/upload/'
    create_directory(upload_directory)

    results_img_directory = '/data/result_images/'
    create_directory(results_img_directory)

    model_directory = '/data/models/'
    create_directory(model_directory)
    

    # could be either 
    # BOTH for both Video and Image
    # IMAGE for Image API Only
    # VIDEO for Video API Only
    # the purpose is to avoid

    api_type = os.getenv('COLORIZER_API_TYPE', 'BOTH')

    image_colorizer = None
    video_colorizer = None
    
    if api_type == 'IMAGE' or api_type == 'BOTH':
        artistic_model_url = 'http://storage.gra5.cloud.ovh.net/v1/AUTH_18b62333a540498882ff446ab602528b/pretrained-models/image/deoldify/ColorizeArtistic_gen.pth'
        get_model_bin(artistic_model_url, os.path.join(model_directory, 'ColorizeArtistic_gen.pth'))
        image_colorizer = get_image_colorizer(artistic=True)

    if api_type == 'VIDEO' or api_type == 'BOTH':
        video_model_url = 'http://storage.gra5.cloud.ovh.net/v1/AUTH_18b62333a540498882ff446ab602528b/pretrained-models/image/deoldify/ColorizeVideo_gen.pth'
        get_model_bin(video_model_url, os.path.join(model_directory, 'ColorizeVideo_gen.pth'))
        video_colorizer = get_video_colorizer()

    
    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True, debug=False)
