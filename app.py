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


# Handle switch between GPU and CPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    del os.environ["CUDA_VISIBLE_DEVICES"]


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# define a predict function as an endpoint
@app.route("/process", methods=["POST"])
def process_image():

    input_path = generate_random_filename(upload_directory,"jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    try:
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)
            try:
                render_factor = request.form.getlist('render_factor')[0]
            except:
                render_factor = 30
            
        else:
            url = request.json["url"]
            download(url, input_path)

            try:
                render_factor = request.json["render_factor"]
            except:
                render_factor = 30

        result = None

        try:
            result = image_colorizer.get_transformed_image(input_path, render_factor=render_factor, post_process=True, watermarked=True)
        except:
            convertToJPG(input_path)
            result = image_colorizer.get_transformed_image(input_path, render_factor=render_factor, post_process=True, watermarked=True)
        finally:
            if result is not None:
                result.save(output_path, quality=95)
                result.close()

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
    global image_colorizer
    global ALLOWED_EXTENSIONS
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    upload_directory = '/data/upload/'
    create_directory(upload_directory)

    results_img_directory = '/data/result_images/'
    create_directory(results_img_directory)

    model_directory = '/data/models/'
    create_directory(model_directory)

    artistic_model_url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"

    # only get the model binay if it not present in /data/models
    get_model_bin(
        artistic_model_url, os.path.join(model_directory, "ColorizeArtistic_gen.pth")
    )

    image_colorizer = get_image_colorizer(artistic=True)

    port = 5000
    host = "0.0.0.0"

    app.run(host=host, port=port, threaded=False)
