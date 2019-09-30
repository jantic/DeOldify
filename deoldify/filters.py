from numpy import ndarray
from abc import ABC, abstractmethod
from .critics import colorize_crit_learner
from fastai.core import *
from fastai.vision import *
from fastai.vision.image import *
from fastai.vision.data import *
from fastai import *
import math
from scipy import misc
import cv2
from PIL import Image as PilImage


class IFilter(ABC):
    @abstractmethod
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        pass


class BaseFilter(IFilter):
    def __init__(self, learn: Learner):
        super().__init__()
        self.learn = learn
        self.norm, self.denorm = normalize_funcs(*imagenet_stats)

    def _transform(self, image: PilImage) -> PilImage:
        return image

    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)

    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        result = self.learn.pred_batch(
            ds_type=DatasetType.Valid, batch=(x[None].cuda(), y[None]), reconstruct=True
        )
        out = result[0]
        out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return PilImage.fromarray(out)

    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, map_to_orig: bool = True):
        super().__init__(learn=learn)
        self.render_base = 16
        self.map_to_orig = map_to_orig

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)

        if self.map_to_orig:
            return self._post_process(model_image, orig_image)
        else:
            return self._post_process(model_image, filtered_image)

    def _transform(self, image: PilImage) -> PilImage:
        return image.convert('LA').convert('RGB')

    # This takes advantage of the fact that human eyes are much less sensitive to
    # imperfections in chrominance compared to luminance.  This means we can
    # save a lot on memory and processing in the model, yet get a great high
    # resolution result at the end.  This is primarily intended just for
    # inference
    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        raw_color = self._unsquare(raw_color, orig)
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
        # do a black and white transform first to get better luminance values
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
        final = PilImage.fromarray(final)
        return final


class MasterFilter(BaseFilter):
    def __init__(self, filters: [IFilter], render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None
    ) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor

        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor)

        return filtered_image
