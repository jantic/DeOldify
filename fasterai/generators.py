from fastai.vision import *
from fastai.vision.learner import cnn_config
from .unet import CustomDynamicUnet, CustomDynamicUnet2
from .loss import FeatureLoss
from .dataset import *

#Weights are implicitly read from ./models/ folder 
def colorize_gen_inference(root_folder:Path, weights_name:str, nf_factor:float)->Learner:
      data = get_dummy_databunch()
      learn = colorize_gen_learner(data=data, gen_loss=F.l1_loss, nf_factor=nf_factor)
      learn.path = root_folder
      learn.load(weights_name)
      learn.model.eval()
      return learn

def colorize_gen_learner(data:ImageDataBunch, gen_loss=FeatureLoss(), arch=models.resnet34, nf_factor:float=1.0)->Learner:
    return custom_unet_learner(data, arch, wd=1e-3, blur=True, norm_type=NormType.Spectral,
                        self_attention=True, y_range=(-3.,3.), loss_func=gen_loss, nf_factor=nf_factor)

#The code below is meant to be merged into fastaiv1 ideally
def custom_unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
                 blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, nf_factor:float=1.0, **kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(CustomDynamicUnet(body, n_classes=data.c, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle, nf_factor=nf_factor), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

#-----------------------------

#Weights are implicitly read from ./models/ folder 
def colorize_gen_inference2(root_folder:Path, weights_name:str, nf_factor:int, arch=models.resnet34)->Learner:
      data = get_dummy_databunch()
      learn = colorize_gen_learner2(data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch)
      learn.path = root_folder
      learn.load(weights_name)
      learn.model.eval()
      return learn

def colorize_gen_learner2(data:ImageDataBunch, gen_loss=FeatureLoss(), arch=models.resnet34, nf_factor:int=1)->Learner:
    return custom_unet_learner2(data, arch=arch, wd=1e-3, blur=True, norm_type=NormType.Spectral,
                        self_attention=True, y_range=(-3.,3.), loss_func=gen_loss, nf_factor=nf_factor)

#The code below is meant to be merged into fastaiv1 ideally
def custom_unet_learner2(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
                 blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, nf_factor:int=1, **kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(CustomDynamicUnet2(body, n_classes=data.c, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle, nf_factor=nf_factor), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn