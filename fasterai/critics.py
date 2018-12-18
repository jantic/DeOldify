from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.vision.gan import *

def colorize_crit_learner(data:ImageDataBunch, loss_critic=AdaptiveLoss(nn.BCEWithLogitsLoss()), nf:int=128)->Learner:
    return Learner(data, gan_critic(nf=nf), metrics=None, loss_func=loss_critic, wd=1e-3)