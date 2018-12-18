from fastai.vision import *
from fastai.vision.models.unet import *
from .loss import FeatureLoss

def colorize_gen_learner(data:ImageDataBunch, gen_loss=FeatureLoss(), arch=models.resnet34):
    return unet_learner(data, arch, wd=1e-3, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=(-3.,3.), loss_func=gen_loss)
