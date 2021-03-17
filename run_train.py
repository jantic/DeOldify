#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

import os
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks.tensorboard import *
from fastai.vision.gan import *
from deoldify.generators import *
from deoldify.critics import *
from deoldify.dataset import *
from deoldify.loss import *
from deoldify.save import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile

#######################################
##  Define your own data folder here ##
#######################################
path = Path('/home/v-yuanqidu/DeOldify/data/pascal_clean_more_val/train/')
path_hr = path
path_lr = path/'bandw'

proj_id = 'ArtisticModel'

gen_name = proj_id + '_gen'
pre_gen_name = gen_name + '_0'
crit_name = proj_id + '_crit'

name_gen = proj_id + '_image_gen'
path_gen = path/name_gen

TENSORBOARD_PATH = Path('data/tensorboard/' + proj_id)

nf_factor = 1.5
pct_start = 1e-8


def get_data(bs:int, sz:int, keep_pct:float):
    return get_colorize_data(sz=sz, bs=bs, crappy_path=path_lr, good_path=path_hr, 
                             random_seed=None, keep_pct=keep_pct)

def get_crit_data(classes, bs, sz):
    src = ImageList.from_folder(path, include=classes, recurse=True).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=sz)
           .databunch(bs=bs).normalize(imagenet_stats))
    return data

def create_training_images(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn).convert('LA').convert('RGB')
    img.save(dest)  
    
def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1
    
def save_gen_images():
    if path_gen.exists(): shutil.rmtree(path_gen)
    path_gen.mkdir(exist_ok=True)
    data_gen = get_data(bs=bs, sz=sz, keep_pct=1.0)
    print(data_gen)
    save_preds(data_gen.fix_dl)
    PIL.Image.open(path_gen.ls()[0])

if not path_lr.exists():
    il = ImageList.from_folder(path_hr)
    parallel(create_training_images, il.items)

bs=1
sz=192
keep_pct=1.0
num_runs = 5

# train generator
data_gen = get_data(bs=bs, sz=sz, keep_pct=keep_pct)
learn_gen = gen_learner_deep(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor)
learn_gen.callback_fns.append(partial(ImageGenTensorboardWriter, base_dir=TENSORBOARD_PATH, name='GenPre'))
learn_gen.fit_one_cycle(5, pct_start=0.8, max_lr=slice(1e-3))
learn_gen.save(pre_gen_name)

# pretrain critic
gc.collect()
data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz)
data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)
learn_critic = colorize_crit_learner(data=data_crit, nf=256)
learn_critic.callback_fns.append(partial(LearnerTensorboardWriter, base_dir=TENSORBOARD_PATH, name='CriticPre'))
learn_critic.fit_one_cycle(4, 1e-4)
learn_critic.save(crit_name)

gc.collect()
data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz)
# learn_crit = colorize_crit_learner(data=data_crit, nf=256).load(crit_new_checkpoint_name, with_opt=False)
# learn_gen = gen_learner_deep(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor).load(gen_old_checkpoint_name, with_opt=False)
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_critic, weights_gen=(1.0,2.0), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.9)), wd=1e-3)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
learn.callback_fns.append(partial(GANTensorboardWriter, base_dir=TENSORBOARD_PATH, name='GanLearner', visual_iters=100))
learn.callback_fns.append(partial(GANSaveCallback, learn_gen=learn_gen, filename=gen_name, save_iters=100))

learn.data = get_data(sz=sz, bs=bs, keep_pct=1.0)
learn_gen.freeze_to(-1)
learn.fit(1,1e-5)
save_gen_images()

path = Path('/home/v-yuanqidu/DeOldify/data/pascal_clean_more_val/val/')
path_hr = path
path_lr = path/'bandw'
if not path_lr.exists():
    il = ImageList.from_folder(path_hr)
    parallel(create_training_images, il.items)
save_gen_images()

