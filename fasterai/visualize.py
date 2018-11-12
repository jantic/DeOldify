from numpy import ndarray
from fastai.torch_imports import *
from fastai.core import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from fastai.dataset import FilesDataset, ImageData, ModelData, open_image
from fastai.transforms import Transform, scale_min, tfms_from_stats, inception_stats
from fastai.transforms import CropType, NoCrop, Denormalize
from .training import GenResult, CriticResult, GANTrainer
from .images import ModelImageSet, EasyTensorImage
from IPython.display import display
from tensorboardX import SummaryWriter
from scipy import misc
import torchvision.utils as vutils
import statistics
from PIL import Image 


class ModelImageVisualizer():
    def __init__(self, default_sz:int=500, results_dir:str=None):
        self.default_sz=default_sz 
        self.denorm = Denormalize(*inception_stats) 
        self.results_dir=None if results_dir is None else Path(results_dir)

    def plot_transformed_image(self, path:str, model:nn.Module, figsize:(int,int)=(20,20), sz:int=None, tfms:[Transform]=[])->ndarray:
        path = Path(path)
        result = self.get_transformed_image_ndarray(path, model, sz, tfms=tfms)
        orig = open_image(str(path))
        fig,axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_image_from_ndarray(orig, axes=axes[0], figsize=figsize)
        self._plot_image_from_ndarray(result, axes=axes[1], figsize=figsize)

        if self.results_dir is not None:
            self._save_result_image(path, result)

    def get_transformed_image_as_pil(self, path:str, model:nn.Module, sz:int=None, tfms:[Transform]=[])->Image:
        path = Path(path)
        array = self.get_transformed_image_ndarray(path, model, sz, tfms=tfms)
        return misc.toimage(array)

    def _save_result_image(self, source_path:Path, result:ndarray):
        result_path = self.results_dir/source_path.name
        misc.imsave(result_path, result)

    def plot_images_from_image_sets(self, image_sets:[ModelImageSet], validation:bool, figsize:(int,int)=(20,20), 
            max_columns:int=6, immediate_display:bool=True):
        num_sets = len(image_sets)
        num_images = num_sets * 2
        rows, columns = self._get_num_rows_columns(num_images, max_columns)

        fig, axes = plt.subplots(rows, columns, figsize=figsize)
        title = 'Validation' if validation else 'Training'
        fig.suptitle(title, fontsize=16)

        for i, image_set in enumerate(image_sets):
            self._plot_image_from_ndarray(image_set.orig.array, axes=axes.flat[i*2])
            self._plot_image_from_ndarray(image_set.gen.array, axes=axes.flat[i*2+1])

        if immediate_display:
            display(fig)

    def get_transformed_image_ndarray(self, path:Path, model:nn.Module, sz:int=None, tfms:[Transform]=[]):
        training = model.training 
        model.eval()
        with torch.no_grad():
            orig = self._get_model_ready_image_ndarray(path, model, sz, tfms)
            orig = VV_(orig[None])
            result = model(orig).detach().cpu().numpy()
            result = self._denorm(result)

        if training:
            model.train()
        return result[0]

    def _denorm(self, image: ndarray):
        if len(image.shape)==3: arr = arr[None]
        return self.denorm(np.rollaxis(image,1,4))

    def _transform(self, orig:ndarray, tfms:[Transform], model:nn.Module, sz:int):
        for tfm in tfms:
            orig,_=tfm(orig, False)
        _,val_tfms = tfms_from_stats(inception_stats, sz, crop_type=CropType.NO, aug_tfms=[])
        val_tfms.tfms = [tfm for tfm in val_tfms.tfms if not isinstance(tfm, NoCrop)]
        orig = val_tfms(orig)
        return orig

    def _get_model_ready_image_ndarray(self, path:Path, model:nn.Module, sz:int=None, tfms:[Transform]=[]):
        im = open_image(str(path))
        sz = self.default_sz if sz is None else sz
        im = scale_min(im, sz)
        im = self._transform(im, tfms, model, sz)
        return im

    def _plot_image_from_ndarray(self, image:ndarray, axes:Axes=None, figsize=(20,20)):
        if axes is None: 
            _,axes = plt.subplots(figsize=figsize)
        clipped_image =np.clip(image,0,1)
        axes.imshow(clipped_image)
        axes.axis('off')

    def _get_num_rows_columns(self, num_images:int, max_columns:int):
        columns = min(num_images, max_columns)
        rows = num_images//columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


class ModelGraphVisualizer():
    def __init__(self):
        return 
     
    def write_model_graph_to_tensorboard(self, ds:FilesDataset, model:nn.Module, tbwriter:SummaryWriter):
        try:
            x,_=ds[0]
            tbwriter.add_graph(model, V(x[None]))
        except Exception as e:
            print(("Failed to generate graph for model: {0}. Note that there's an outstanding issue with "
                + "scopes being addressed here:  https://github.com/pytorch/pytorch/pull/12400").format(e))

class ModelHistogramVisualizer():
    def __init__(self):
        return 

    def write_tensorboard_histograms(self, model:nn.Module, iter_count:int, tbwriter:SummaryWriter):
        for name, param in model.named_parameters():
            tbwriter.add_histogram('/weights/' + name, param, iter_count)
    


class ModelStatsVisualizer(): 
    def __init__(self):
        return 

    def write_tensorboard_stats(self, model:nn.Module, iter_count:int, tbwriter:SummaryWriter):
        gradients = [x.grad  for x in model.parameters() if x.grad is not None]
        gradient_nps = [to_np(x.data) for x in gradients]
 
        if len(gradients) == 0:
            return 

        avg_norm = sum(x.data.norm() for x in gradients)/len(gradients)
        tbwriter.add_scalar('/gradients/avg_norm', avg_norm, iter_count)

        median_norm = statistics.median(x.data.norm() for x in gradients)
        tbwriter.add_scalar('/gradients/median_norm', median_norm, iter_count)

        max_norm = max(x.data.norm() for x in gradients)
        tbwriter.add_scalar('/gradients/max_norm', max_norm, iter_count)

        min_norm = min(x.data.norm() for x in gradients)
        tbwriter.add_scalar('/gradients/min_norm', min_norm, iter_count)

        num_zeros = sum((np.asarray(x)==0.0).sum() for x in  gradient_nps)
        tbwriter.add_scalar('/gradients/num_zeros', num_zeros, iter_count)


        avg_gradient= sum(x.data.mean() for x in gradients)/len(gradients)
        tbwriter.add_scalar('/gradients/avg_gradient', avg_gradient, iter_count)

        median_gradient = statistics.median(x.data.median() for x in gradients)
        tbwriter.add_scalar('/gradients/median_gradient', median_gradient, iter_count)

        max_gradient = max(x.data.max() for x in gradients) 
        tbwriter.add_scalar('/gradients/max_gradient', max_gradient, iter_count)

        min_gradient = min(x.data.min() for x in gradients) 
        tbwriter.add_scalar('/gradients/min_gradient', min_gradient, iter_count)

class ImageGenVisualizer():
    def __init__(self):
        self.model_vis = ModelImageVisualizer()

    def output_image_gen_visuals(self, md:ImageData, model:nn.Module, iter_count:int, tbwriter:SummaryWriter, jupyter:bool=False):
        self._output_visuals(ds=md.val_ds, model=model, iter_count=iter_count, tbwriter=tbwriter, jupyter=jupyter, validation=True)
        self._output_visuals(ds=md.trn_ds, model=model, iter_count=iter_count, tbwriter=tbwriter, jupyter=jupyter, validation=False)

    def _output_visuals(self, ds:FilesDataset, model:nn.Module, iter_count:int, tbwriter:SummaryWriter, 
            validation:bool, jupyter:bool=False):
        #TODO:  Parameterize these
        start_idx=0
        count = 8
        end_index = start_idx + count
        idxs = list(range(start_idx,end_index))
        image_sets = ModelImageSet.get_list_from_model(ds=ds, model=model, idxs=idxs)
        self._write_tensorboard_images(image_sets=image_sets, iter_count=iter_count, tbwriter=tbwriter, validation=validation)
        if jupyter:
            self._show_images_in_jupyter(image_sets, validation=validation)
    
    def _write_tensorboard_images(self, image_sets:[ModelImageSet], iter_count:int, tbwriter:SummaryWriter, validation:bool):
        orig_images = []
        gen_images = []
        real_images = []

        for image_set in image_sets:
            orig_images.append(image_set.orig.tensor)
            gen_images.append(image_set.gen.tensor)
            real_images.append(image_set.real.tensor)

        prefix = 'val' if validation else 'train'

        tbwriter.add_image(prefix + ' orig images', vutils.make_grid(orig_images, normalize=True), iter_count)
        tbwriter.add_image(prefix + ' gen images', vutils.make_grid(gen_images, normalize=True), iter_count)
        tbwriter.add_image(prefix + ' real images', vutils.make_grid(real_images, normalize=True), iter_count)


    def _show_images_in_jupyter(self, image_sets:[ModelImageSet], validation:bool):
        #TODO:  Parameterize these
        figsize=(20,20)
        max_columns=4
        immediate_display=True
        self.model_vis.plot_images_from_image_sets(image_sets, figsize=figsize, max_columns=max_columns, 
            immediate_display=immediate_display, validation=validation)


class GANTrainerStatsVisualizer():
    def __init__(self):
        return

    def write_tensorboard_stats(self, gresult:GenResult, cresult:CriticResult, iter_count:int, tbwriter:SummaryWriter):
        tbwriter.add_scalar('/loss/hingeloss', cresult.hingeloss, iter_count)
        tbwriter.add_scalar('/loss/dfake', cresult.dfake, iter_count)
        tbwriter.add_scalar('/loss/dreal', cresult.dreal, iter_count)
        tbwriter.add_scalar('/loss/gcost', gresult.gcost, iter_count)
        tbwriter.add_scalar('/loss/gcount', gresult.iters, iter_count)
        tbwriter.add_scalar('/loss/gaddlloss', gresult.gaddlloss, iter_count)

    def print_stats_in_jupyter(self, gresult:GenResult, cresult:CriticResult):
        print(f'\nHingeLoss {cresult.hingeloss}; RScore {cresult.dreal}; FScore {cresult.dfake}; GAddlLoss {gresult.gaddlloss}; ' + 
                f'Iters: {gresult.iters}; GCost: {gresult.gcost};')


class LearnerStatsVisualizer():
    def __init__(self):
        return
    
    def write_tensorboard_stats(self, metrics, iter_count:int, tbwriter:SummaryWriter):
        if isinstance(metrics, list):
            tbwriter.add_scalar('/loss/trn_loss', metrics[0], iter_count)    
            if len(metrics) == 1: return
            tbwriter.add_scalar('/loss/val_loss', metrics[1], iter_count)        
            if len(metrics) == 2: return

            for metric in metrics[2:]:
                name = metric.__name__
                tbwriter.add_scalar('/loss/'+name, metric, iter_count)
                    
        else: 
            tbwriter.add_scalar('/loss/trn_loss', metrics, iter_count)

  





