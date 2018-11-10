from fastai.dataset import FilesDataset, ModelData, ImageData, Transform, RandomFlip, RandomZoom, TfmType
from fastai.dataset import get_cv_idxs, split_by_idx, tfms_from_stats, inception_stats, open_image
from fastai.core import *


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames:np.array, y:np.array, transforms:[Transform], path:Path, x_tfms:[Transform]=[]):
        self.y=y
        self.x_tfms=x_tfms
        assert(len(fnames)==len(y))
        super().__init__(fnames, transforms, path)
    def get_x(self, i): 
        x = super().get_x(i)
        for tfm in self.x_tfms:
            x,_ = tfm(x, False)
        return x
    def get_y(self, i): 
        return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): 
        return 0 

class ImageGenDataLoader():
    def __init__(self, sz:int, bs:int, path:Path, random_seed:int=None, keep_pct:float=1.0, x_tfms:[Transform]=[], 
            file_exts=('jpg','jpeg','png'), extra_aug_tfms:[Transform]=[], reduce_x_scale:int=1):
        
        self.md = None
        self.sz = sz
        self.bs = bs 
        self.path = path
        self.x_tfms = x_tfms
        self.random_seed = random_seed
        self.keep_pct = keep_pct
        self.file_exts = file_exts
        self.extra_aug_tfms=extra_aug_tfms
        self.reduce_x_scale=reduce_x_scale

    def get_model_data(self):
        if self.md is not None:
            return self.md

        resize_amt = self._get_resize_amount()
        resize_folder = 'tmp'
        ((val_x,trn_x),(val_y,trn_y)) = self._get_filename_sets(resize_folder)

        if len(trn_x) == 0:
            raise ValueError('No image files were found in specified image directory. Path provided was: ' + str(self.path))

        aug_tfms = [RandomFlip(tfm_y=TfmType.PIXEL), RandomZoom(zoom_max=0.18, tfm_y=TfmType.PIXEL)] 
        aug_tfms.extend(self.extra_aug_tfms)
        sz_x = self.sz//self.reduce_x_scale
        sz_y = self.sz
        tfms = (tfms_from_stats(inception_stats, sz=sz_x, sz_y=sz_y, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms))
        dstype = MatchedFilesDataset
        datasets = ImageData.get_ds(dstype, (trn_x,trn_y), (val_x,val_y), tfms, path=self.path, x_tfms=self.x_tfms)
        resize_path = os.path.join(self.path,resize_folder,str(resize_amt))
        self.md = self._load_model_data(resize_folder, resize_path, resize_amt, datasets, trn_x)
        return self.md

    def _load_model_data(self, resize_folder:str, resize_path:str, resize_amt:int, datasets, trn_x):
        #optimization
        if os.path.exists(os.path.join(resize_path,trn_x[0])):
            return ImageData(Path(resize_path), datasets, self.bs, num_workers=16, classes=None)
        
        md = ImageData(self.path.parent, datasets, self.bs, num_workers=16, classes=None)
        if resize_amt != self.sz: 
            md = md.resize(resize_amt, new_path=str(resize_folder))

        return md

    def _get_filename_sets(self, resize_folder:str):
        exclude_str = '/' + resize_folder + '/'
        paths = self._find_files_recursively(self.path,self.file_exts) 
        paths = filter(lambda path: not re.search(exclude_str, str(path)), paths)
        fnames_full = [Path(str(fname).replace(str(self.path) + '/','')) for fname in paths] 
        self._update_np_random_seed()
        keeps = np.random.rand(len(fnames_full)) < self.keep_pct
        fnames = np.array(fnames_full, copy=False)[keeps]
        val_idxs = get_cv_idxs(len(fnames), val_pct=min(0.01/self.keep_pct, 0.1))
        return split_by_idx(val_idxs, np.array(fnames), np.array(fnames))

    def _update_np_random_seed(self):
        if self.random_seed is None:
            np.random.seed()
        else:
            np.random.seed(self.random_seed)

    def _get_resize_amount(self):
        if self.sz<96:
            return 128
        if self.sz <192:
            return 256
        return self.sz

    
    def _find_files_recursively(self, root_path:Path, extensions:(str)):
        matches = []
        for root, dirnames, filenames in os.walk(str(root_path)):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    matches.append(os.path.join(root, filename))
        return matches


    