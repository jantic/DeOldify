from numpy import ndarray
from abc import ABC, abstractmethod
from .generators import Unet34, GeneratorModule
from .transforms import BlackAndWhiteTransform
from fastai.torch_imports import *
from fastai.core import *
from fastai.transforms import Transform, scale_min, tfms_from_stats, inception_stats
from fastai.transforms import CropType, NoCrop, Denormalize, Scale, scale_to
import math
from scipy import misc

class Padding():
    def __init__(self, top:int, bottom:int, left:int, right:int):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
  
class Filter(ABC):
    def __init__(self, tfms:[Transform]):
        super().__init__()
        self.tfms=tfms
        self.denorm = Denormalize(*inception_stats)
    
    @abstractmethod
    def filter(self, orig_image:ndarray, render_factor:int)->ndarray:
        pass

    def _transform(self, orig:ndarray, sz:int):
        for tfm in self.tfms:
            orig,_=tfm(orig, False)
        _,val_tfms = tfms_from_stats(inception_stats, sz, crop_type=CropType.NO, aug_tfms=[])
        val_tfms.tfms = [tfm for tfm in val_tfms.tfms if not (isinstance(tfm, NoCrop) or isinstance(tfm, Scale))]
        orig = val_tfms(orig)
        return orig

    def _scale_to_square(self, orig:ndarray, targ:int, interpolation=cv2.INTER_AREA):
        r,c,*_ = orig.shape
        ratio = targ/max(r,c)
        #a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        #I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
        sz = (targ, targ)
        return cv2.resize(orig, sz, interpolation=interpolation)

    def _get_model_ready_image_ndarray(self, orig:ndarray, sz:int):
        result = self._scale_to_square(orig, sz)
        sz=result.shape[0]
        result = self._transform(result, sz)
        return result

    def _denorm(self, image: ndarray):
        if len(image.shape)==3: 
            image = image[None]
        return self.denorm(np.rollaxis(image,1,4))

    def _model_process(self, model:GeneratorModule, orig:ndarray, sz:int):
        orig = self._get_model_ready_image_ndarray(orig, sz)
        orig = VV_(orig[None]) 
        result = model(orig)
        result = result.detach().cpu().numpy()
        result = self._denorm(result)
        return result[0]

    def _convert_to_pil(self, im_array:ndarray):
        im_array = np.clip(im_array,0,1)
        return misc.toimage(im_array)


class Colorizer(Filter):
    def __init__(self, gpu:int, weights_path:Path):
        super().__init__(tfms=[BlackAndWhiteTransform()])
        self.model = Unet34(nf_factor=2).cuda(gpu)
        load_model(self.model, weights_path)
        self.model.eval()
        torch.no_grad()
        self.render_base = 32
    
    def filter(self, orig_image:ndarray, render_factor:int=14)->ndarray:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(self.model, orig=orig_image, sz=render_sz)
        return self._post_process(model_image, orig_image)


    #This takes advantage of the fact that human eyes are much less sensitive to 
    #imperfections in chrominance compared to luminance.  This means we can
    #save a lot on memory and processing in the model, yet get a great high
    #resolution result at the end.  This is primarily intended just for 
    #inference
    def _post_process(self, raw_color:ndarray, orig:ndarray):
        for tfm in self.tfms:
            orig,_=tfm(orig, False)

        sz = (orig.shape[1], orig.shape[0])
        raw_color = cv2.resize(raw_color, sz, interpolation=cv2.INTER_AREA)
        color_yuv = cv2.cvtColor(raw_color, cv2.COLOR_BGR2YUV)
        #do a black and white transform first to get better luminance values
        orig_yuv = cv2.cvtColor(orig, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:,:,1:3] = color_yuv[:,:,1:3]
        return cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)    