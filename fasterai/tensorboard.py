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

    def write_tensorboard_histograms(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model'):
        try:
            for param_name, param in model.named_parameters():
                tbwriter.add_histogram(name + '/weights/' + param_name, param, iteration)
        except Exception as e:
            print(("Failed to update histogram for model:  {0}").format(e))


class ModelStatsVisualizer(): 
    def __init__(self):
        return 

    def write_tensorboard_stats(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model_stats'):
        try:
            gradients = [x.grad  for x in model.parameters() if x.grad is not None]
            gradient_nps = [to_np(x.data) for x in gradients]
    
            if len(gradients) == 0:
                return 

            avg_norm = sum(x.data.norm() for x in gradients)/len(gradients)
            tbwriter.add_scalar(name + '/gradients/avg_norm', avg_norm, iteration)

            median_norm = statistics.median(x.data.norm() for x in gradients)
            tbwriter.add_scalar(name + '/gradients/median_norm', median_norm, iteration)

            max_norm = max(x.data.norm() for x in gradients)
            tbwriter.add_scalar(name + '/gradients/max_norm', max_norm, iteration)

            min_norm = min(x.data.norm() for x in gradients)
            tbwriter.add_scalar(name + '/gradients/min_norm', min_norm, iteration)

            num_zeros = sum((np.asarray(x)==0.0).sum() for x in  gradient_nps)
            tbwriter.add_scalar(name + '/gradients/num_zeros', num_zeros, iteration)


            avg_gradient= sum(x.data.mean() for x in gradients)/len(gradients)
            tbwriter.add_scalar(name + '/gradients/avg_gradient', avg_gradient, iteration)

            median_gradient = statistics.median(x.data.median() for x in gradients)
            tbwriter.add_scalar(name + '/gradients/median_gradient', median_gradient, iteration)

            max_gradient = max(x.data.max() for x in gradients) 
            tbwriter.add_scalar(name + '/gradients/max_gradient', max_gradient, iteration)

            min_gradient = min(x.data.min() for x in gradients) 
            tbwriter.add_scalar(name + '/gradients/min_gradient', min_gradient, iteration)
        except Exception as e:
            print(("Failed to update tensorboard stats for model:  {0}").format(e))

class ImageGenVisualizer():
    def output_image_gen_visuals(self, learn:Learner, trn_batch:Tuple, val_batch:Tuple, iteration:int, tbwriter:SummaryWriter):
        self._output_visuals(learn=learn, batch=val_batch, iteration=iteration, tbwriter=tbwriter, ds_type=DatasetType.Valid)
        self._output_visuals(learn=learn, batch=trn_batch, iteration=iteration, tbwriter=tbwriter, ds_type=DatasetType.Train)

    def _output_visuals(self, learn:Learner, batch:Tuple, iteration:int, tbwriter:SummaryWriter, ds_type: DatasetType):
        image_sets = ModelImageSet.get_list_from_model(learn=learn, batch=batch, ds_type=ds_type)
        self._write_tensorboard_images(image_sets=image_sets, iteration=iteration, tbwriter=tbwriter, ds_type=ds_type)
    
    def _write_tensorboard_images(self, image_sets:[ModelImageSet], iteration:int, tbwriter:SummaryWriter, ds_type: DatasetType):
        try:
            orig_images = []
            gen_images = []
            real_images = []

            for image_set in image_sets:
                orig_images.append(image_set.orig.px)
                gen_images.append(image_set.gen.px)
                real_images.append(image_set.real.px)

            prefix = str(ds_type)

            tbwriter.add_image(prefix + ' orig images', vutils.make_grid(orig_images, normalize=True), iteration)
            tbwriter.add_image(prefix + ' gen images', vutils.make_grid(gen_images, normalize=True), iteration)
            tbwriter.add_image(prefix + ' real images', vutils.make_grid(real_images, normalize=True), iteration)
        except Exception as e:
            print(("Failed to update tensorboard images for model:  {0}").format(e))


#--------Below are what you actually want ot use, in practice----------------#

class LearnerTensorboardWriter(LearnerCallback):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, weight_iters:int=1000, stats_iters:int=1000):
        super().__init__(learn=learn)
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.loss_iters = loss_iters
        self.weight_iters = weight_iters
        self.stats_iters = stats_iters
        self.weight_vis = ModelHistogramVisualizer()
        self.model_vis = ModelStatsVisualizer() 
        self.data = None
        self.metrics_root = '/metrics/'

    def _update_batches_if_needed(self):
        #one_batch function is extremely slow.  this is an optimization
        update_batches = self.data is not self.learn.data

        if update_batches:
            self.data = self.learn.data
            self.trn_batch = self.learn.data.one_batch(DatasetType.Train, detach=True, denorm=False, cpu=False)
            self.val_batch = self.learn.data.one_batch(DatasetType.Valid, detach=True, denorm=False, cpu=False)

    def _write_model_stats(self, iteration):
        self.model_vis.write_tensorboard_stats(model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter) 

    def _write_training_loss(self, iteration, last_loss):
        trn_loss = to_np(last_loss)
        self.tbwriter.add_scalar(self.metrics_root + 'train_loss', trn_loss, iteration)

    def _write_weight_histograms(self, iteration):
        self.weight_vis.write_tensorboard_histograms(model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)


    def _write_metrics(self, iteration, last_metrics, start_idx:int=2):
        recorder = self.learn.recorder

        for i, name in enumerate(recorder.names[start_idx:]):
            if len(last_metrics) < i+1: return 
            value = last_metrics[i]
            self.tbwriter.add_scalar(self.metrics_root + name, value, iteration)  
  
    def on_batch_end(self, last_loss, metrics, iteration, **kwargs):
        if iteration==0: return
        self._update_batches_if_needed()

        if iteration % self.loss_iters == 0: 
            self._write_training_loss(iteration, last_loss)

        if iteration % self.weight_iters == 0:
            self._write_weight_histograms(iteration)

        if iteration % self.stats_iters == 0:
            self._write_model_stats(iteration)

    def on_epoch_end(self, metrics, last_metrics, iteration, **kwargs):
        self._write_metrics(iteration, last_metrics)


class GANTensorboardWriter(LearnerTensorboardWriter):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, weight_iters:int=1000, 
                stats_iters:int=1000, visual_iters:int=100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters, 
                        weight_iters=weight_iters, stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageGenVisualizer()

    #override
    def _write_weight_histograms(self, iteration):
        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic
        self.weight_vis.write_tensorboard_histograms(model=generator, iteration=iteration, tbwriter=self.tbwriter, name='generator')
        self.weight_vis.write_tensorboard_histograms(model=critic, iteration=iteration, tbwriter=self.tbwriter, name='critic')

    #override
    def _write_model_stats(self, iteration):
        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic
        self.model_vis.write_tensorboard_stats(model=generator, iteration=iteration, tbwriter=self.tbwriter, name='gen_model_stats')
        self.model_vis.write_tensorboard_stats(model=critic, iteration=iteration, tbwriter=self.tbwriter, name='crit_model_stats')

    def _write_images(self, iteration):
        trainer = self.learn.gan_trainer
        recorder = trainer.recorder
        gen_mode = trainer.gen_mode
        trainer.switch(gen_mode=True)
        self.img_gen_vis.output_image_gen_visuals(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, 
                                               iteration=iteration, tbwriter=self.tbwriter)
        trainer.switch(gen_mode=gen_mode)

    def on_batch_end(self, metrics, iteration, **kwargs):
        super().on_batch_end(metrics=metrics, iteration=iteration, **kwargs)
        if iteration==0: return
        if iteration % self.visual_iters == 0:
            self._write_images(iteration)

              

class ImageGenTensorboardWriter(LearnerTensorboardWriter):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, weight_iters:int=1000, 
                stats_iters:int=1000, visual_iters:int=100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters, weight_iters=weight_iters, 
                        stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageGenVisualizer()

    def _write_images(self, iteration):
        self.img_gen_vis.output_image_gen_visuals(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, 
            iteration=iteration, tbwriter=self.tbwriter)

    def on_batch_end(self, metrics, iteration, **kwargs):
        super().on_batch_end(metrics=metrics, iteration=iteration, **kwargs)

        if iteration==0:
            return

        if iteration % self.visual_iters == 0:
            self._write_images(iteration)
