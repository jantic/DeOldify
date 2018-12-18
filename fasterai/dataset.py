import fastai
from fastai import *
from fastai.core import *
from fastai.vision import *


def get_colorize_data(sz:int, bs:int, crappy_path:Path, good_path:Path, random_seed:int=None, 
        keep_pct:float=1.0, num_workers:int=8)->ImageDataBunch:

    src = (ImageImageList.from_folder(crappy_path)
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .random_split_by_pct(0.1, seed=random_seed))

    data = (src.label_from_func(lambda x: good_path/x.relative_to(crappy_path))
        #TODO:  Revisit transforms used here....
        .transform(get_transforms(), size=sz, tfm_y=True)
        .databunch(bs=bs, num_workers=num_workers)
        .normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
