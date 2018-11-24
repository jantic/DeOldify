from fastai.core import *
from fastai.conv_learner import model_meta, cut_model
from fastai.transforms import scale_min
from .modules import ConvBlock, UnetBlock, UpSampleBlock, SaveFeatures
from abc import ABC, abstractmethod
from torchvision import transforms
from torch.nn.utils.spectral_norm import spectral_norm

class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def set_trainable(self, trainable:bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self, precompute:bool=False)->[]:
        pass

    @abstractmethod
    def forward(self, x_in:torch.Tensor, max_render_sz:int=400):
        pass

    def freeze_to(self, n:int):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def get_device(self):
        return next(self.parameters()).device


class AbstractUnet(GeneratorModule): 
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__()
        assert (math.log(scale,2)).is_integer()
        self.rn, self.lr_cut = self._get_pretrained_resnet_base()
        ups = self._get_decoding_layers(nf_factor=nf_factor, scale=scale)
        self.relu = nn.ReLU()
        self.up1 = ups[0]
        self.up2 = ups[1]
        self.up3 = ups[2]
        self.up4 = ups[3]
        self.up5 = ups[4]
        self.out= nn.Sequential(ConvBlock(32*nf_factor, 3, ks=3, actn=False, bn=False, sn=True), nn.Tanh())

    @abstractmethod
    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        pass

    @abstractmethod
    def _get_decoding_layers(self, nf_factor:int, scale:int):
        pass

    #Gets around irritating inconsistent halving coming from resnet
    def _pad(self, x:torch.Tensor, target:torch.Tensor, total_padh:int, total_padw:int)-> torch.Tensor:
        h = x.shape[2] 
        w = x.shape[3]

        target_h = target.shape[2]*2
        target_w = target.shape[3]*2

        if h<target_h or w<target_w:
            padh = target_h-h if target_h > h else 0
            total_padh = total_padh + padh
            padw = target_w-w if target_w > w else 0
            total_padw = total_padw + padw
            return (F.pad(x, (0,padw,0,padh), "reflect",0), total_padh, total_padw)

        return (x, total_padh, total_padw)

    def _remove_padding(self, x:torch.Tensor, padh:int, padw:int)->torch.Tensor:
        if padw == 0 and padh == 0:
            return x 
        
        target_h = x.shape[2]-padh
        target_w = x.shape[3]-padw
        return x[:,:,:target_h, :target_w]

    def _encode(self, x:torch.Tensor):
        x = self.rn[0](x)
        x = self.rn[1](x)
        x = self.rn[2](x)
        enc0 = x
        x = self.rn[3](x)
        x = self.rn[4](x)
        enc1 = x
        x = self.rn[5](x)
        enc2 = x
        x = self.rn[6](x)
        enc3 = x
        x = self.rn[7](x)
        return (x, enc0, enc1, enc2, enc3)

    def _decode(self, x:torch.Tensor, enc0:torch.Tensor, enc1:torch.Tensor, enc2:torch.Tensor, enc3:torch.Tensor):
        padh = 0
        padw = 0
        x = self.relu(x)
        enc3, padh, padw = self._pad(enc3, x, padh, padw)
        x = self.up1(x, enc3)
        enc2, padh, padw  = self._pad(enc2, x, padh, padw)
        x = self.up2(x, enc2)
        enc1, padh, padw  = self._pad(enc1, x, padh, padw)
        x = self.up3(x, enc1)
        enc0, padh, padw  = self._pad(enc0, x, padh, padw)
        x = self.up4(x, enc0)
        #This is a bit too much padding being removed, but I 
        #haven't yet figured out a good way to determine what 
        #exactly should be removed.  This is consistently more 
        #than enough though.
        x = self.up5(x)
        x = self.out(x)
        x = self._remove_padding(x, padh, padw)
        return x

    def forward(self, x:torch.Tensor):
        x, enc0, enc1, enc2, enc3 = self._encode(x)
        x = self._decode(x, enc0, enc1, enc2, enc3)
        return x
    
    def get_layer_groups(self, precompute:bool=False)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]
    
    def close(self):
        for sf in self.sfs: 
            sf.remove()


class Unet34(AbstractUnet): 
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        f = resnet34
        cut,lr_cut = model_meta[f]
        cut-=layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor:int, scale:int):
        self_attention=True
        bn=True
        sn=True
        leakyReLu=False
        layers = []
        layers.append(UnetBlock(512,256,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,128,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,512*nf_factor, sn=sn, self_attention=self_attention, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UpSampleBlock(256*nf_factor, 32*nf_factor, 2*scale, sn=sn, leakyReLu=leakyReLu, bn=bn))
        return layers 


class Unet101(AbstractUnet): 
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        f = resnet101
        cut,lr_cut = model_meta[f]
        cut-=layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor:int, scale:int):
        self_attention=True
        bn=True
        sn=True
        leakyReLu=False
        layers = []
        layers.append(UnetBlock(2048,1024,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,512,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,256,512*nf_factor, sn=sn, self_attention=self_attention, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UpSampleBlock(256*nf_factor, 32*nf_factor, 2*scale, sn=sn, leakyReLu=leakyReLu, bn=bn))
        return layers 

