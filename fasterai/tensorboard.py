import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from fastai.core import *
import statistics
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

class ModelImageSet():
    @staticmethod
    def get_list_from_model(learn:Learner, ds_type:DatasetType, batch:Tuple)->[]:
        image_sets = []
        x,y = batch[0],batch[1]
        preds = learn.pred_batch(ds_type=ds_type, batch=(x,y), reconstruct=True)
        
        for orig_px, real_px, gen in zip(x,y,preds):
            orig = Image(px=orig_px)
            real = Image(px=real_px)
            image_set = ModelImageSet(orig=orig, real=real, gen=gen)
            image_sets.append(image_set)

        return image_sets  

    def __init__(self, orig:Image, real:Image, gen:Image):
        self.orig = orig
        self.real = real
        self.gen = gen

#TODO:  There aren't any callbacks using this yet.  Not sure if we want this included (not sure if it's useful, honestly)
class ModelGraphVisualizer():
    def __init__(self):
        return

    def write_model_graph_to_tensorboard(self, md:DataBunch, model:nn.Module, tbwriter:SummaryWriter):
        x,y = md.one_batch(ds_type=DatasetType.Valid, detach=False, denorm=False)
        tbwriter.add_graph(model=model, input_to_model=x)


class ModelHistogramVisualizer():
    def __init__(self):
        return

    def write_tensorboard_histograms(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model'):
        for param_name, values in model.named_parameters():
            tag = name + '/weights/' + param_name
            tbwriter.add_histogram(tag=tag, values=values, global_step=iteration)


class ModelStatsVisualizer():
    def __init__(self):
        self.gradients_root = '/gradients/'

    def write_tensorboard_stats(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model_stats'):
        gradients = [x.grad for x in model.parameters() if x.grad is not None]
        gradient_nps = [to_np(x.data) for x in gradients]

        if len(gradients) == 0: return

        avg_norm = sum(x.data.norm() for x in gradients)/len(gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'avg_norm', scalar_value=avg_norm, global_step=iteration)

        median_norm = statistics.median(x.data.norm() for x in gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'median_norm', scalar_value=median_norm, global_step=iteration)

        max_norm = max(x.data.norm() for x in gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'max_norm', scalar_value=max_norm, global_step=iteration)

        min_norm = min(x.data.norm() for x in gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'min_norm', scalar_value=min_norm, global_step=iteration)

        num_zeros = sum((np.asarray(x) == 0.0).sum() for x in gradient_nps)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'num_zeros', scalar_value=num_zeros, global_step=iteration)

        avg_gradient = sum(x.data.mean() for x in gradients)/len(gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'avg_gradient', scalar_value=avg_gradient, global_step=iteration)

        median_gradient = statistics.median(x.data.median() for x in gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'median_gradient', scalar_value=median_gradient, global_step=iteration)

        max_gradient = max(x.data.max() for x in gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'max_gradient', scalar_value=max_gradient, global_step=iteration)

        min_gradient = min(x.data.min() for x in gradients)
        tbwriter.add_scalar(
            tag=name + self.gradients_root + 'min_gradient', scalar_value=min_gradient, global_step=iteration)


class ImageGenVisualizer():
    def output_image_gen_visuals(self, learn:Learner, trn_batch:Tuple, val_batch:Tuple, iteration:int, tbwriter:SummaryWriter):
        self._output_visuals(learn=learn, batch=val_batch, iteration=iteration,
                             tbwriter=tbwriter, ds_type=DatasetType.Valid)
        self._output_visuals(learn=learn, batch=trn_batch, iteration=iteration,
                             tbwriter=tbwriter, ds_type=DatasetType.Train)

    def _output_visuals(self, learn:Learner, batch:Tuple, iteration:int, tbwriter:SummaryWriter, ds_type:DatasetType):
        image_sets = ModelImageSet.get_list_from_model(
            learn=learn, batch=batch, ds_type=ds_type)
        self._write_tensorboard_images(
            image_sets=image_sets, iteration=iteration, tbwriter=tbwriter, ds_type=ds_type)

    def _write_tensorboard_images(self, image_sets:[ModelImageSet], iteration:int, tbwriter:SummaryWriter, ds_type:DatasetType):
        orig_images = []
        gen_images = []
        real_images = []

        for image_set in image_sets:
            orig_images.append(image_set.orig.px)
            gen_images.append(image_set.gen.px)
            real_images.append(image_set.real.px)

        prefix = ds_type.name

        tbwriter.add_image(
            tag=prefix + ' orig images', img_tensor=vutils.make_grid(orig_images, normalize=True), global_step=iteration)
        tbwriter.add_image(
            tag=prefix + ' gen images', img_tensor=vutils.make_grid(gen_images, normalize=True), global_step=iteration)
        tbwriter.add_image(
            tag=prefix + ' real images', img_tensor=vutils.make_grid(real_images, normalize=True), global_step=iteration)


#--------Below are what you actually want to use, in practice----------------#

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
        # one_batch function is extremely slow.  this is an optimization
        update_batches = self.data is not self.learn.data

        if update_batches:
            self.data = self.learn.data
            self.trn_batch = self.learn.data.one_batch(
                ds_type=DatasetType.Train, detach=True, denorm=False, cpu=False)
            self.val_batch = self.learn.data.one_batch(
                ds_type=DatasetType.Valid, detach=True, denorm=False, cpu=False)

    def _write_model_stats(self, iteration:int):
        self.model_vis.write_tensorboard_stats(
            model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)

    def _write_training_loss(self, iteration:int, last_loss:Tensor):
        scalar_value = to_np(last_loss)
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_weight_histograms(self, iteration:int):
        self.weight_vis.write_tensorboard_histograms(
            model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)

    #TODO:  Relying on a specific hardcoded start_idx here isn't great.  Is there a better solution?
    def _write_metrics(self, iteration:int, last_metrics:MetricsList, start_idx:int=2):
        recorder = self.learn.recorder

        for i, name in enumerate(recorder.names[start_idx:]):
            if len(last_metrics) < i+1: return
            scalar_value = last_metrics[i]
            tag = self.metrics_root + name
            self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def on_batch_end(self, last_loss:Tensor, iteration:int, **kwargs):
        if iteration == 0: return
        self._update_batches_if_needed()

        if iteration % self.loss_iters == 0:
            self._write_training_loss(iteration=iteration, last_loss=last_loss)

        if iteration % self.weight_iters == 0:
            self._write_weight_histograms(iteration=iteration)

    # Doing stuff here that requires gradient info, because they get zeroed out afterwards in training loop
    def on_backward_end(self, iteration:int, **kwargs):
        if iteration == 0: return
        self._update_batches_if_needed()

        if iteration % self.stats_iters == 0:
            self._write_model_stats(iteration=iteration)

    def on_epoch_end(self, last_metrics:MetricsList, iteration:int, **kwargs):
        self._write_metrics(iteration=iteration, last_metrics=last_metrics)

# TODO:  We're overriding almost everything here.  Seems like a good idea to question that ("is a" vs "has a")
class GANTensorboardWriter(LearnerTensorboardWriter):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, weight_iters:int=1000,
                 stats_iters:int=1000, visual_iters:int=100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters,
                         weight_iters=weight_iters, stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageGenVisualizer()
        self.gen_stats_updated = True
        self.crit_stats_updated = True

    # override
    def _write_weight_histograms(self, iteration:int):
        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic
        self.weight_vis.write_tensorboard_histograms(
            model=generator, iteration=iteration, tbwriter=self.tbwriter, name='generator')
        self.weight_vis.write_tensorboard_histograms(
            model=critic, iteration=iteration, tbwriter=self.tbwriter, name='critic')

    # override
    def _write_model_stats(self, iteration:int):
        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic

        # Don't want to write stats when model is not iterated on and hence has zeroed out gradients
        gen_mode = trainer.gen_mode

        if gen_mode and not self.gen_stats_updated:
            self.model_vis.write_tensorboard_stats(
                model=generator, iteration=iteration, tbwriter=self.tbwriter, name='gen_model_stats')
            self.gen_stats_updated = True

        if not gen_mode and not self.crit_stats_updated:
            self.model_vis.write_tensorboard_stats(
                model=critic, iteration=iteration, tbwriter=self.tbwriter, name='crit_model_stats')
            self.crit_stats_updated = True

    # override
    def _write_training_loss(self, iteration:int, last_loss:Tensor):
        trainer = self.learn.gan_trainer
        recorder = trainer.recorder

        if len(recorder.losses) > 0:
            scalar_value = to_np((recorder.losses[-1:])[0])
            tag = self.metrics_root + 'train_loss'
            self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_images(self, iteration:int):
        trainer = self.learn.gan_trainer
        #TODO:  Switching gen_mode temporarily seems a bit hacky here.  Certainly not a good side-effect.  Is there a better way?
        gen_mode = trainer.gen_mode

        try:
            trainer.switch(gen_mode=True)
            self.img_gen_vis.output_image_gen_visuals(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch,
                                                    iteration=iteration, tbwriter=self.tbwriter)
        finally:                                      
            trainer.switch(gen_mode=gen_mode)

    # override
    def on_batch_end(self, iteration:int, **kwargs):
        super().on_batch_end(iteration=iteration, **kwargs)
        if iteration == 0: return
        if iteration % self.visual_iters == 0:
            self._write_images(iteration=iteration)

    # override
    def on_backward_end(self, iteration:int, **kwargs):
        if iteration == 0: return
        self._update_batches_if_needed()

        #TODO:  This could perhaps be implemented as queues of requests instead but that seemed like overkill. 
        # But I'm not the biggest fan of maintaining these boolean flags either... Review pls.
        if iteration % self.stats_iters == 0:
            self.gen_stats_updated = False
            self.crit_stats_updated = False

        if not (self.gen_stats_updated and self.crit_stats_updated):
            self._write_model_stats(iteration=iteration)


class ImageGenTensorboardWriter(LearnerTensorboardWriter):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, weight_iters:int=1000,
                 stats_iters: int = 1000, visual_iters: int = 100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters, weight_iters=weight_iters,
                         stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageGenVisualizer()

    def _write_images(self, iteration:int):
        self.img_gen_vis.output_image_gen_visuals(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch,
                                                  iteration=iteration, tbwriter=self.tbwriter)

    # override
    def on_batch_end(self, iteration:int, **kwargs):
        super().on_batch_end(iteration=iteration, **kwargs)

        if iteration == 0: return

        if iteration % self.visual_iters == 0:
            self._write_images(iteration=iteration)
