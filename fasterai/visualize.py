from fastai.core import *
from fastai.vision import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .filters import IFilter, MasterFilter, ColorizerFilter
from .generators import colorize_gen_inference
from IPython.display import display
from tensorboardX import SummaryWriter
from scipy import misc
from PIL import Image 


class ModelImageVisualizer():
    def __init__(self, filter:IFilter, results_dir:str=None):
        self.filter = filter
        self.results_dir=None if results_dir is None else Path(results_dir)

    def _open_pil_image(self, path:Path)->Image:
        return PIL.Image.open(path).convert('RGB')

    def plot_transformed_image(self, path:str, figsize:(int,int)=(20,20), render_factor:int=None)->Image:
        path = Path(path)
        result = self.get_transformed_image(path, render_factor)
        orig = self._open_pil_image(path)
        fig,axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_image(orig, axes=axes[0], figsize=figsize)
        self._plot_image(result, axes=axes[1], figsize=figsize)

        if self.results_dir is not None:
            self._save_result_image(path, result)

    def _save_result_image(self, source_path:Path, image:Image):
        result_path = self.results_dir/source_path.name
        image.save(result_path)

    def get_transformed_image(self, path:Path, render_factor:int=None)->Image:
        orig_image = self._open_pil_image(path)
        filtered_image = self.filter.filter(orig_image, orig_image, render_factor=render_factor)
        return filtered_image

    def _plot_image(self, image:Image, axes:Axes=None, figsize=(20,20)):
        if axes is None: 
            _,axes = plt.subplots(figsize=figsize)
        axes.imshow(np.asarray(image)/255)
        axes.axis('off')

    def _get_num_rows_columns(self, num_images:int, max_columns:int)->(int,int):
        columns = min(num_images, max_columns)
        rows = num_images//columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


def get_colorize_visualizer(root_folder:Path=Path('./'), weights_name:str='colorize_gen', 
        results_dir = 'result_images', nf_factor:float=1.25, render_factor:int=21)->ModelImageVisualizer:
    learn = colorize_gen_inference(root_folder=root_folder, weights_name=weights_name, nf_factor=nf_factor)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis





