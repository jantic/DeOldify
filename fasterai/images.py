import numpy as np
from fastai.torch_imports import *
from fastai.core import *
from fastai.dataset import FilesDataset
from pathlib import Path
from itertools import repeat
from PIL import Image
from numpy import ndarray
from datetime import datetime

class EasyTensorImage():
    def __init__(self, source_tensor:torch.Tensor, ds:FilesDataset):
        self.array = self._convert_to_denormed_ndarray(source_tensor, ds=ds)   
        self.tensor = self._convert_to_denormed_tensor(self.array)
    
    def _convert_to_denormed_ndarray(self, raw_tensor:torch.Tensor, ds:FilesDataset):
        raw_array = raw_tensor.clone().data.cpu().numpy()
        if raw_array.shape[1] != 3:
            array = np.zeros((3, 1, 1))
            return array
        else:
            return ds.denorm(raw_array)[0]

    def _convert_to_denormed_tensor(self, denormed_array: ndarray):
        return V(np.moveaxis(denormed_array,2,0))

class ModelImageSet():
    @staticmethod
    def get_list_from_model(ds:FilesDataset, model:nn.Module, idxs:[int]):
        image_sets = []
        training = model.training
        model.eval()
        
        for idx in idxs:
            x,y=ds[idx]
            orig_tensor = VV(x[None]) 
            real_tensor = V(y[None])
            gen_tensor = model(orig_tensor)
            gen_easy = EasyTensorImage(gen_tensor, ds)
            orig_easy = EasyTensorImage(orig_tensor, ds)
            real_easy = EasyTensorImage(real_tensor, ds)
            image_set = ModelImageSet(orig_easy,real_easy,gen_easy)
            image_sets.append(image_set)

        if training:
            model.train()

        return image_sets  

    def __init__(self, orig:EasyTensorImage, real:EasyTensorImage, gen:EasyTensorImage):
        self.orig=orig
        self.real=real
        self.gen=gen