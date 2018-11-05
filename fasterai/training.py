from fastai.core import *
from fastai.torch_imports import *
from fastai.dataset import Transform
from fastai.layer_optimizer import LayerOptimizer
from fastai.sgdr import CircularLR_beta
from fasterai.modules import ConvBlock
from fasterai.generators import GeneratorModule
from fasterai.dataset import ImageGenDataLoader
from collections import Iterable
import torch.utils.hooks as hooks
from abc import ABC
 

class CriticModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def freeze_to(self, n:int):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)
    
    def set_trainable(self, trainable:bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self)->[]:
        pass

    def get_device(self):
        return next(self.parameters()).device

class DCCritic(CriticModule):

    def _generate_reduce_layers(self, nf:int):
        layers=[]
        layers.append(nn.Dropout2d(0.5))
        layers.append(ConvBlock(nf, nf*2, 4, 2, bn=False, sn=True, leakyReLu=True, self_attention=True))
        return layers

    def __init__(self, ni:int=3, nf:int=128):
        super().__init__()
        scale:int=16
        sn=True
        self_attention=True
        assert (math.log(scale,2)).is_integer()
        self.initial = nn.Sequential(
            ConvBlock(ni, nf, 4, 2, bn=False, sn=sn, leakyReLu=True),
            nn.Dropout2d(0.2),
            ConvBlock(nf, nf, 3, 1, bn=False, sn=sn, leakyReLu=True)
        )

        cndf = nf
        mid_layers =  []  

        for i in range(int(math.log(scale,2))-1):
            use_attention = (i == 0 and self_attention) 
            layers = self._generate_reduce_layers(nf=cndf)
            mid_layers.extend(layers)
            cndf = int(cndf*2)
   
        self.mid = nn.Sequential(*mid_layers)
        out_layers=[]
        out_layers.append(ConvBlock(cndf, 1, ks=4, stride=1, bias=False, bn=False, sn=sn, pad=0, actn=False))
        self.out = nn.Sequential(*out_layers) 

    def get_layer_groups(self)->[]:
        return children(self)
    
    def forward(self, input):
        x=self.initial(input)
        x=self.mid(x)
        return self.out(x), x


class GenResult():
    def __init__(self, gcost: np.array, iters: int, gaddlloss: np.array):
        self.gcost=gcost
        self.iters=iters
        self.gaddlloss=gaddlloss

class CriticResult():
    def __init__(self, hingeloss: np.array, dreal: np.array, dfake: np.array, dcost: np.array):
        self.hingeloss=hingeloss
        self.dreal=dreal
        self.dfake=dfake
        self.dcost=dcost


class GANTrainSchedule(): 
    @staticmethod
    def generate_schedules(szs:[int], bss:[int], path:Path, keep_pcts:[float], save_base_name:str, 
        c_lrs:[float], g_lrs:[float], gen_freeze_tos:[int], lrs_unfreeze_factor:float=0.1, 
        x_noise:int=None, random_seed=None, x_tfms:[Transform]=[], extra_aug_tfms:[Transform]=[],
        reduce_x_scale=1):

        scheds = []

        for i in range(len(szs)):
            sz = szs[i] 
            bs = bss[i]
            keep_pct = keep_pcts[i]
            gen_freeze_to = gen_freeze_tos[i]
            critic_lrs = c_lrs * (lrs_unfreeze_factor if gen_freeze_to == 0 else 1.0)
            gen_lrs = g_lrs * (lrs_unfreeze_factor if gen_freeze_to == 0 else 1.0)
            critic_save_path = path.parent/(save_base_name + '_critic_' + str(sz) + '.h5')
            gen_save_path = path.parent/(save_base_name + '_gen_' + str(sz) + '.h5')
            sched = GANTrainSchedule(sz=sz, bs=bs, path=path, critic_lrs=critic_lrs, gen_lrs=gen_lrs,
                critic_save_path=critic_save_path, gen_save_path=gen_save_path, random_seed=random_seed,
                x_noise=x_noise, keep_pct=keep_pct, x_tfms=x_tfms, extra_aug_tfms=extra_aug_tfms,  
                reduce_x_scale=reduce_x_scale, gen_freeze_to=gen_freeze_to)
            scheds.append(sched)
        
        return scheds


    def __init__(self, sz:int, bs:int, path:Path, critic_lrs:[float], gen_lrs:[float],
            critic_save_path: Path, gen_save_path: Path, random_seed=None, x_noise:int=None, 
            keep_pct:float=1.0, num_epochs=1, x_tfms:[Transform]=[], extra_aug_tfms:[Transform]=[], 
            reduce_x_scale=1, gen_freeze_to=0):
        self.md = None

        self.data_loader = ImageGenDataLoader(sz=sz, bs=bs, path=path, random_seed=random_seed, x_noise=x_noise,
            keep_pct=keep_pct, x_tfms=x_tfms, extra_aug_tfms=extra_aug_tfms, reduce_x_scale=reduce_x_scale)
        self.sz = sz
        self.bs = bs
        self.path = path
        self.critic_lrs = np.array(critic_lrs)
        self.gen_lrs = np.array(gen_lrs)
        self.critic_save_path = critic_save_path
        self.gen_save_path = gen_save_path
        self.num_epochs=num_epochs
        self.gen_freeze_to = gen_freeze_to
        
    #Lazy init
    def get_model_data(self):
        return self.data_loader.get_model_data()

class GANTrainer():
    def __init__(self, netD: nn.Module, netG: GeneratorModule, save_iters:int=1000, genloss_fns:[]=[]):
        self.netD = netD
        self.netG = netG
        self._train_loop_hooks = OrderedDict()
        self._train_begin_hooks = OrderedDict()
        self.genloss_fns = genloss_fns
        self.save_iters=save_iters
        self.iters = 0

    def register_train_loop_hook(self, hook):
        handle = hooks.RemovableHandle(self._train_loop_hooks)
        self._train_loop_hooks[handle.id] = hook
        return handle

    def register_train_begin_hook(self, hook):
        handle = hooks.RemovableHandle(self._train_begin_hooks)
        self._train_begin_hooks[handle.id] = hook
        return handle


    def train(self, scheds:[GANTrainSchedule]):
        for sched in scheds:
            self.md = sched.get_model_data()   
            self.dpath = sched.critic_save_path
            self.gpath = sched.gen_save_path
            epochs = sched.num_epochs   
            lrs_gen = sched.gen_lrs
            lrs_critic = sched.critic_lrs

            if self.iters == 0:
                self.gen_sched = self._generate_clr_sched(self.netG, use_clr_beta=(1,8), lrs=lrs_gen, cycle_len=1)
                self.critic_sched = self._generate_clr_sched(self.netD, use_clr_beta=(1,8), lrs=lrs_critic, cycle_len=1)
                self._call_train_begin_hooks()
            else:
                self.gen_sched.init_lrs = lrs_gen
                self.critic_sched.init_lrs = lrs_critic
            
            self._get_inner_module(self.netG).freeze_to(sched.gen_freeze_to)
            self.critic_sched.on_train_begin()
            self.gen_sched.on_train_begin()
        
            for epoch in trange(epochs):
                self._train_one_epoch()

    def _get_inner_module(self, model:nn.Module):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _generate_clr_sched(self, model:nn.Module, use_clr_beta: (int), lrs: [float], cycle_len: int):
        wds = 1e-7
        opt_fn = partial(optim.Adam, betas=(0.0,0.9))
        layer_opt = LayerOptimizer(opt_fn, self._get_inner_module(model).get_layer_groups(), lrs, wds)
        div,pct = use_clr_beta[:2]
        moms = use_clr_beta[2:] if len(use_clr_beta) > 3 else None
        cycle_end =  None
        return CircularLR_beta(layer_opt, len(self.md.trn_dl)*cycle_len, on_cycle_end=cycle_end, div=div, pct=pct, momentums=moms)

    def _train_one_epoch(self)->int:
        self.netD.train()
        self.netG.train()
        data_iter = iter(self.md.trn_dl)
        n = len(self.md.trn_dl)

        with tqdm(total=n) as pbar:
            while True:
                self.iters+=1
                cresult = self._train_critic(data_iter, pbar)
                
                if cresult is None:
                    break
                
                gresult = self._train_generator(data_iter, pbar, cresult)

                if gresult is None:
                    break

                self._save_if_applicable()
                self._call_train_loop_hooks(gresult, cresult)
    
    def _call_train_begin_hooks(self):
        for hook in self._train_begin_hooks.values():
            hook_result = hook()
            if hook_result is not None:
                raise RuntimeError(
                    "train begin hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))

    def _call_train_loop_hooks(self, gresult: GenResult, cresult: CriticResult):
        for hook in self._train_loop_hooks.values():
            hook_result = hook(gresult, cresult)
            if hook_result is not None:
                raise RuntimeError(
                    "train loop hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))

    def _get_next_training_images(self, data_iter: Iterable)->(torch.Tensor,torch.Tensor):
        x, y = next(data_iter, (None, None))

        if x is None or y is None:
            return (None,None)

        orig_image = V(x)
        real_image = V(y) 
        return (orig_image, real_image)


    def _train_critic(self, data_iter: Iterable, pbar: tqdm)->CriticResult:
        self._get_inner_module(self.netD).set_trainable(True)
        self._get_inner_module(self.netG).set_trainable(False)
        orig_image, real_image = self._get_next_training_images(data_iter)
        if orig_image is None:
            return None

        cresult = self._train_critic_once(orig_image, real_image)
        pbar.update()
        return cresult

    def _train_critic_once(self, orig_image: torch.Tensor, real_image: torch.Tensor)->CriticResult:                     
        fake_image = self.netG(orig_image)
        dfake_raw,_ = self.netD(fake_image)
        dfake = torch.nn.ReLU()(1.0+dfake_raw).mean()
        dreal_raw,_ = self.netD(real_image)
        dreal = torch.nn.ReLU()(1.0-dreal_raw).mean()
        self.netD.zero_grad()     
        hingeloss = dfake + dreal
        hingeloss.backward()
        self.critic_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(hingeloss))
        self.gen_sched.on_batch_end(to_np(-dfake))
        return CriticResult(to_np(hingeloss), to_np(dreal), to_np(dfake), to_np(hingeloss))

    def _train_generator(self, data_iter: Iterable, pbar: tqdm, cresult: CriticResult)->GenResult:
        orig_image, real_image = self._get_next_training_images(data_iter)   
        if orig_image is None:
            return None

        gresult = self._train_generator_once(orig_image, real_image, cresult)  
        pbar.update() 
        return gresult

    def _train_generator_once(self, orig_image: torch.Tensor, real_image: torch.Tensor, 
            cresult: CriticResult)->GenResult:
        self._get_inner_module(self.netD).set_trainable(False)
        self._get_inner_module(self.netG).set_trainable(True)
        self.netG.zero_grad()         
        fake_image = self.netG(orig_image)
        gcost = -self._get_dscore(fake_image)
        gaddlloss = self._calc_addl_gen_loss(real_image, fake_image) 
        total_loss = gcost if gaddlloss is None else gcost + gaddlloss
        total_loss.backward()
        self.gen_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(cresult.dcost))
        self.gen_sched.on_batch_end(to_np(gcost))
        return GenResult(to_np(gcost), self.iters, to_np(gaddlloss))


    def _save_if_applicable(self):
        if self.iters % self.save_iters == 0:
            save_model(self.netD, self.dpath)
            save_model(self.netG, self.gpath)

    def _get_dscore(self, new_image: torch.Tensor):
        scores, _ = self.netD(new_image)
        return scores.mean()
    

    def _calc_addl_gen_loss(self, real_data: torch.Tensor, fake_data: torch.Tensor)->torch.Tensor:
        total_loss = V(0.0)
        for loss_fn in self.genloss_fns:
            loss = loss_fn(fake_data, real_data)
            total_loss = total_loss + loss
        return total_loss