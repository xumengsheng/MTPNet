import cv2
import numpy as np
import torch
from torch import abs_, nn
from torch import optim
from PIL import Image
from typing import Mapping,Sequence,Tuple,Union
from torchvision.models import vgg19
from torch.nn import functional as F
import torchvision.models.vgg as vgg
from guided_filter_pytorch.guided_filter import GuidedFilter
from basicsr.utils.registry import LOSS_REGISTRY
_reduction_modes = ['none', 'mean', 'sum']

class L_Abs_sideout(nn.Module):
    def __init__(self):
        super(L_Abs_sideout, self).__init__()
        self.resolution_weight=[1.,1.,1.,1.]

    def forward(self,x,flare_gt):
        #[256,256],[128,128],[64,64],[32,32]
        Abs_loss=0
        for i in range(4):
            flare_loss=torch.abs(x[i]-flare_gt[i])
            Abs_loss+=torch.mean(flare_loss)*self.resolution_weight[i]
        return Abs_loss
    

class L_Abs(nn.Module):
    def __init__(self):
        super(L_Abs, self).__init__()

    def forward(self,x,flare_gt,base_gt,mask_gt,merge_gt):
        base_predicted=base_gt*mask_gt+(1-mask_gt)*x
        flare_predicted=merge_gt-(1-mask_gt)*x
        base_loss=torch.abs(base_predicted-base_gt)
        flare_loss=torch.abs(flare_predicted-flare_gt)
        Abs_loss=torch.mean(base_loss+flare_loss)
        return Abs_loss

@LOSS_REGISTRY.register()
class L_Abs_pure(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_pure, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss)
        return self.loss_weight*Abs_loss

@LOSS_REGISTRY.register()
class L_Abs_weighted(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_weighted, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt,weight):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss*weight)
        '''
        mask_area=torch.mean(torch.abs(weight))
        if mask_area>0:
            return self.loss_weight*Abs_loss/mask_area
        else:
        '''
        return self.loss_weight*Abs_loss

@LOSS_REGISTRY.register()
class L_percepture(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_percepture, self).__init__()
        self.loss_weight=loss_weight
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model=model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.selected_feature_index=[2,7,12,21,30]
        self.layer_weight=[1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
    
    def extract_feature(self,x):
        selected_features = []
        for i,model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, source, target):
        source_feature = self.extract_feature(source)
        target_feature = self.extract_feature(target)
        len_feature=len(source_feature)
        perceptual_loss=0
        for i in range(len_feature):
            perceptual_loss+=self.mae_loss(source_feature[i],target_feature[i])*self.layer_weight[i]
        return self.loss_weight*perceptual_loss

@LOSS_REGISTRY.register()
class CorssEntropy(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CorssEntropy, self).__init__()
        self.loss_weight=loss_weight
        self.loss = nn.BCELoss()

    def forward(self, source, target):
        cross_entropy_loss = self.loss(source, target)
        return self.loss_weight*cross_entropy_loss

@LOSS_REGISTRY.register()
class WeightedBCE(nn.Module):
    def __init__(self, loss_weight=1.0,class_weight=[1.0,1.0]):
        super(WeightedBCE, self).__init__()
        self.loss_weight=loss_weight
        self.class_weight = class_weight

    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - self.class_weight[1] * target * torch.log(input) - (1 - target) * self.class_weight[0] * torch.log(1 - input)
        return torch.mean(bce)



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
@LOSS_REGISTRY.register()
class ContrastLoss(nn.Module):
    def __init__(self,loss_weight,ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.loss_weight=loss_weight

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss*self.loss_weight


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

@LOSS_REGISTRY.register()
class HFLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(HFLoss, self).__init__()
        self.loss_weight = loss_weight
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        _,pred=get_LFHF(pred)
        _,target=get_LFHF(target)

        loss=self.l1(pred,target)*self.loss_weight

        return  loss




def get_LFHF(image, rad_list=[4, 8, 16, 32], eps_list=[0.001, 0.0001]):
    def decomposition(guide, inp, rad_list, eps_list):
        LF_list = []
        HF_list = []
        for radius in rad_list:
            for eps in eps_list:
                gf = GuidedFilter(radius, eps)
                LF = gf(guide, inp)
                LF[LF > 1] = 1
                LF_list.append(LF)
                HF_list.append(inp - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF

    image = torch.clamp(image, min=0.0, max=1.0)
    # Compute the LF-HF features of the image
    img_lf, img_hf = decomposition(guide=image,
                                       inp=image,
                                       rad_list=rad_list,
                                       eps_list=eps_list)
    return img_lf, img_hf


@LOSS_REGISTRY.register()
class APILoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(APILoss, self).__init__()
        self.loss_weight = loss_weight
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        loss=adaptive_pixel_intensity_loss(pred,target)*self.loss_weight

        return  loss

def adaptive_pixel_intensity_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))

    return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()