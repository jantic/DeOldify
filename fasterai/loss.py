from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.callbacks  import hook_outputs
import torchvision.models as models


class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[5,15,2]):
        super().__init__()
        self.base_loss = F.l1_loss
        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [i-1 for i,o in enumerate(children(self.m_feat)) if isinstance(o,nn.MaxPool2d)]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def _gram_matrix(self, x:torch.Tensor):
        n,c,h,w = x.size()
        x = x.view(n, c, -1)
        return (x @ x.transpose(1,2))/(c*h*w)

    def make_features(self, x:torch.Tensor, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input:torch.Tensor, target:torch.Tensor):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input,target)]
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [self.base_loss(self._gram_matrix(f_in), self._gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): 
        self.hooks.remove()


