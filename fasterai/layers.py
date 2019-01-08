from fastai.layers import *
from fastai.torch_core import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable


#The code below is meant to be merged into fastaiv1 ideally

def conv_layer2(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, is_1d:bool=False,
               norm_type:Optional[NormType]=NormType.Batch,  use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False,
               extra_bn:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn==True
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    
    #TODO:  Account for 1D
    #if norm_type==NormType.Weight: layers.append(MeanOnlyBatchNorm(nf))

    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.running_mean.zero_()
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, inp):
        size = list(inp.size())
        gamma = self.weight.view(1, self.num_features, 1, 1)
        beta = self.bias.view(1, self.num_features, 1, 1)

        if self.training:
            avg = torch.mean(inp.view(size[0], self.num_features, -1), dim=2)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * torch.mean(avg.data, dim=0)
        else:
            avg = Variable(self.running_mean.repeat(size[0], 1), requires_grad=False)

        output = inp - avg.view(size[0], size[1], 1, 1)
        output = output*gamma + beta

        return output