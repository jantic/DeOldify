import numpy as np
from fastai.core import *
from fastai.vision import *
from pathlib import Path
from itertools import repeat
from PIL import Image as PilImage
from numpy import ndarray
from datetime import datetime
from fastai.vision.image import *


class ModelImageSet():
    @staticmethod
    def get_list_from_model(learn: Learner, ds_type: DatasetType, batch:Tuple)->[]:
        image_sets = []
        x,y = batch[0],batch[1]
        #x,y = learn.data.one_batch(ds_type, detach=False, denorm=False)
        preds = learn.pred_batch(ds_type=ds_type, batch=(x,y), reconstruct=True)
        
        for orig,real,gen_image in zip(x,y,preds):
            orig_image = Image(orig)
            real_image = Image(real)
            image_set = ModelImageSet(orig_image, real_image, gen_image)
            image_sets.append(image_set)

        return image_sets  

    def __init__(self, orig:Image, real:Image, gen:Image):
        self.orig=orig
        self.real=real
        self.gen=gen