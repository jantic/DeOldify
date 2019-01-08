from fastai.vision import *
from fastai.vision.learner import cnn_config
from fasterai.unet import *
from .loss import FeatureLoss

def colorize_gen_learner(data:ImageDataBunch, gen_loss=FeatureLoss(), arch=models.resnet34):
    return unet_learner2(data, arch, wd=1e-3, blur=True, norm_type=NormType.Spectral,
                        self_attention=True, y_range=(-3.,3.), loss_func=gen_loss)

#The code below is meant to be merged into fastaiv1 ideally

def unet_learner2(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
                 blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, **kwargs:Any)->None:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(DynamicUnet2(body, n_classes=data.c, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn
