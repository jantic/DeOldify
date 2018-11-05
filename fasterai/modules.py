from fastai.torch_imports import *
from fastai.conv_learner import *
from torch.nn.utils.spectral_norm import spectral_norm


class ConvBlock(nn.Module):
    def __init__(self, ni:int, no:int, ks:int=3, stride:int=1, pad:int=None, actn:bool=True, 
            bn:bool=True, bias:bool=True, sn:bool=False, leakyReLu:bool=False, self_attention:bool=False,
            inplace_relu:bool=True):
        super().__init__()   
        if pad is None: pad = ks//2//stride

        if sn:
            layers = [spectral_norm(nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias))]
        else:
            layers = [nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias)]
        if actn:
            layers.append(nn.LeakyReLU(0.2, inplace=inplace_relu)) if leakyReLu else layers.append(nn.ReLU(inplace=inplace_relu)) 
        if bn:
            layers.append(nn.BatchNorm2d(no))
        if self_attention:
            layers.append(SelfAttention(no, 1))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class UpSampleBlock(nn.Module):
    @staticmethod
    def _conv(ni:int, nf:int, ks:int=3, bn:bool=True, sn:bool=False, leakyReLu:bool=False):
        layers = [ConvBlock(ni, nf, ks=ks, sn=sn, bn=bn, actn=False, leakyReLu=leakyReLu)]
        return nn.Sequential(*layers)

    @staticmethod
    def _icnr(x:torch.Tensor, scale:int=2):
        init=nn.init.kaiming_normal_
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def __init__(self, ni:int, nf:int, scale:int=2, ks:int=3, bn:bool=True, sn:bool=False, leakyReLu:bool=False):
        super().__init__()
        layers = []
        assert (math.log(scale,2)).is_integer()

        for i in range(int(math.log(scale,2))):
            layers += [UpSampleBlock._conv(ni, nf*4,ks=ks, bn=bn, sn=sn, leakyReLu=leakyReLu), 
                nn.PixelShuffle(2)]
            if bn:
                layers += [nn.BatchNorm2d(nf)]

            ni = nf
                       
        self.sequence = nn.Sequential(*layers)
        self._icnr_init()
        
    def _icnr_init(self):
        conv_shuffle = self.sequence[0][0].seq[0]
        kernel = UpSampleBlock._icnr(conv_shuffle.weight)
        conv_shuffle.weight.data.copy_(kernel)
    
    def forward(self, x):
        return self.sequence(x)


class UnetBlock(nn.Module):
    def __init__(self, up_in:int , x_in:int , n_out:int, bn:bool=True, sn:bool=False, leakyReLu:bool=False, 
            self_attention:bool=False, inplace_relu:bool=True):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = ConvBlock(x_in,  x_out,  ks=1, bn=False, actn=False, sn=sn, inplace_relu=inplace_relu)
        self.tr_conv = UpSampleBlock(up_in, up_out, 2, bn=bn, sn=sn, leakyReLu=leakyReLu)
        self.relu = nn.LeakyReLU(0.2, inplace=inplace_relu) if leakyReLu else nn.ReLU(inplace=inplace_relu)
        out_layers = []
        if bn: 
            out_layers.append(nn.BatchNorm2d(n_out))
        if self_attention:
            out_layers.append(SelfAttention(n_out))
        self.out = nn.Sequential(*out_layers)
        
        
    def forward(self, up_p:int, x_p:int):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        x = torch.cat([up_p,x_p], dim=1)
        x = self.relu(x)
        return self.out(x)
        return out

class SaveFeatures():
    features=None
    def __init__(self, m:nn.Module): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def remove(self): 
        self.hook.remove()

class SelfAttention(nn.Module):
    def __init__(self, in_channel:int, gain:int=1):
        super().__init__()
        self.query = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.key = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.value = self._spectral_init(nn.Conv1d(in_channel, in_channel, 1), gain=gain)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _spectral_init(self, module:nn.Module, gain:int=1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return spectral_norm(module)

    def forward(self, input:torch.Tensor):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out