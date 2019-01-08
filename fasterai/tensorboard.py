import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from fastai.core import *
import statistics
from .images import ModelImageSet
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class ModelGraphVisualizer():
    def __init__(self):
        return 
     
    def write_model_graph_to_tensorboard(self, md:DataBunch, model:nn.Module, tbwriter:SummaryWriter):
        try:
            x,y = md.one_batch(DatasetType.Valid, detach=False, denorm=False)
            tbwriter.add_graph(model, x)
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
    def output_image_gen_visuals(self, learn:Learner, trn_batch:Tuple, val_batch:Tuple, iter_count:int, tbwriter:SummaryWriter):
        self._output_visuals(learn=learn, batch=val_batch, iter_count=iter_count, tbwriter=tbwriter, ds_type=DatasetType.Valid)
        self._output_visuals(learn=learn, batch=trn_batch, iter_count=iter_count, tbwriter=tbwriter, ds_type=DatasetType.Train)

    def _output_visuals(self, learn:Learner, batch:Tuple, iter_count:int, tbwriter:SummaryWriter, ds_type: DatasetType):
        image_sets = ModelImageSet.get_list_from_model(learn=learn, batch=batch, ds_type=ds_type)
        self._write_tensorboard_images(image_sets=image_sets, iter_count=iter_count, tbwriter=tbwriter, ds_type=ds_type)
    
    def _write_tensorboard_images(self, image_sets:[ModelImageSet], iter_count:int, tbwriter:SummaryWriter, ds_type: DatasetType):
        orig_images = []
        gen_images = []
        real_images = []

        for image_set in image_sets:
            orig_images.append(image_set.orig.px)
            gen_images.append(image_set.gen.px)
            real_images.append(image_set.real.px)

        prefix = str(ds_type)

        tbwriter.add_image(prefix + ' orig images', vutils.make_grid(orig_images, normalize=True), iter_count)
        tbwriter.add_image(prefix + ' gen images', vutils.make_grid(gen_images, normalize=True), iter_count)
        tbwriter.add_image(prefix + ' real images', vutils.make_grid(real_images, normalize=True), iter_count)


#--------Below are what you actually want ot use, in practice----------------#

class ModelTensorboardStatsWriter():
    def __init__(self, base_dir: Path, module: nn.Module, name: str, stats_iters: int=10):
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.hook = module.register_forward_hook(self.forward_hook)
        self.stats_iters = stats_iters
        self.iter_count = 0
        self.model_vis = ModelStatsVisualizer() 

    def forward_hook(self, module:nn.Module, input, output): 
        self.iter_count += 1
        if self.iter_count % self.stats_iters == 0:
            self.model_vis.write_tensorboard_stats(module, iter_count=self.iter_count, tbwriter=self.tbwriter)  


    def close(self):
        self.tbwriter.close()
        self.hook.remove()

class GANTensorboardWriter(LearnerCallback):
    def __init__(self, learn:Learner, base_dir:Path, name:str, stats_iters:int=10, 
            visual_iters:int=200, weight_iters:int=1000):
        super().__init__(learn=learn)
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.stats_iters = stats_iters
        self.visual_iters = visual_iters
        self.weight_iters = weight_iters
        self.img_gen_vis = ImageGenVisualizer()
        self.graph_vis = ModelGraphVisualizer()
        self.weight_vis = ModelHistogramVisualizer()
        self.data = None

    def on_batch_end(self, iteration, metrics, **kwargs):
        if iteration==0:
            return

        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic
        recorder = trainer.recorder
        #one_batch is extremely slow.  this is an optimization
        update_batches = self.data is not self.learn.data

        if update_batches:
            self.data = self.learn.data
            self.trn_batch = self.learn.data.one_batch(DatasetType.Train, detach=False, denorm=False)
            self.val_batch = self.learn.data.one_batch(DatasetType.Valid, detach=False, denorm=False)

        if iteration % self.stats_iters == 0:  
            if len(recorder.losses) > 0:      
                trn_loss = to_np((recorder.losses[-1:])[0])
                self.tbwriter.add_scalar('/loss/trn_loss', trn_loss, iteration)

            if len(recorder.val_losses) > 0:
                val_loss = (recorder.val_losses[-1:])[0]
                self.tbwriter.add_scalar('/loss/val_loss', val_loss, iteration) 

            #TODO:  Figure out how to do metrics here and gan vs critic loss
            #values = [met[-1:] for met in recorder.metrics]

        if iteration % self.visual_iters == 0:
            gen_mode = trainer.gen_mode
            trainer.switch(gen_mode=True)
            self.img_gen_vis.output_image_gen_visuals(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, 
                                                    iter_count=iteration, tbwriter=self.tbwriter)
            trainer.switch(gen_mode=gen_mode)

        if iteration % self.weight_iters == 0:
            self.weight_vis.write_tensorboard_histograms(model=generator, iter_count=iteration, tbwriter=self.tbwriter)
            self.weight_vis.write_tensorboard_histograms(model=critic, iter_count=iteration, tbwriter=self.tbwriter)
              


class ImageGenTensorboardWriter(LearnerCallback):
    def __init__(self, learn:Learner, base_dir:Path, name:str, stats_iters:int=25, 
            visual_iters:int=200, weight_iters:int=25):
        super().__init__(learn=learn)
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.stats_iters = stats_iters
        self.visual_iters = visual_iters
        self.weight_iters = weight_iters
        self.iter_count = 0
        self.weight_vis = ModelHistogramVisualizer()
        self.img_gen_vis = ImageGenVisualizer()
        self.data = None

    def on_batch_end(self, iteration, last_loss, metrics, **kwargs):
        if iteration==0:
            return

        #one_batch is extremely slow.  this is an optimization
        update_batches = self.data is not self.learn.data

        if update_batches:
            self.data = self.learn.data
            self.trn_batch = self.learn.data.one_batch(DatasetType.Train, detach=False, denorm=False)
            self.val_batch = self.learn.data.one_batch(DatasetType.Valid, detach=False, denorm=False)

        if iteration % self.stats_iters == 0: 
            trn_loss = to_np(last_loss)
            self.tbwriter.add_scalar('/loss/trn_loss', trn_loss, iteration)

        if iteration % self.visual_iters == 0:
            self.img_gen_vis.output_image_gen_visuals(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, 
                iter_count=iteration, tbwriter=self.tbwriter)

        if iteration % self.weight_iters == 0:
            self.weight_vis.write_tensorboard_histograms(model=self.learn.model, iter_count=iteration, tbwriter=self.tbwriter)

    def on_epoch_end(self, iteration, metrics, last_metrics, **kwargs):  
        #TODO: Not a fan of this indexing but...what to do?
        val_loss = last_metrics[0]
        self.tbwriter.add_scalar('/loss/val_loss', val_loss, iteration)   