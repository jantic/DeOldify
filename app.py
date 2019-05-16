# import the necessary packages
import os
import sys
import requests
import ssl
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file

from uuid import uuid4

from os import path
import torch

import fastai
from fasterai.visualize import *
from pathlib import Path


torch.backends.cudnn.benchmark=True

colorizer = get_image_colorizer(artistic=True)

os.environ['CUDA_VISIBLE_DEVICES']='0'

app = Flask(__name__)

# define a predict function as an endpoint
@app.route("/process", methods=["POST"])
def process():
    source_url = request.json["source_url"]
    render_factor = int(request.json["render_factor"])

    upload_directory = 'upload'
    if not os.path.exists(upload_directory):
           os.mkdir(upload_directory)

    random_filename = str(uuid4()) + '.png'

    colorizer.plot_transformed_image_from_url(url=source_url, path=os.path.join(upload_directory, random_filename), figsize=(20,20),
            render_factor=render_factor, display_render_factor=True, compare=False)

    callback = send_file(os.path.join("result_images", random_filename), mimetype='image/jpeg')

    os.remove(os.path.join("result_images", random_filename))
    os.remove(os.path.join("upload", random_filename))

    return callback


if __name__ == '__main__':
    port = 5000
    host = '0.0.0.0'
    app.run(host=host, port=port, threaded=True)
