import cv2
import numpy as np
import torch
from torch import abs_, nn
from torch import optim
from PIL import Image
from typing import Mapping,Sequence,Tuple,Union
from torchvision.models import vgg19
from torchvision import transforms
from kornia.losses import MS_SSIMLoss
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from basicsr.utils.registry import LOSS_REGISTRY
from guided_filter_pytorch.guided_filter import GuidedFilter
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
class MaskLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss, self).__init__()
        self.loss_weight=loss_weight
        self.loss = nn.BCELoss()
        self.to_gray = transforms.Grayscale(num_output_channels=1)
    def forward(self, source, target):
        flare = self.to_gray(target)
        one = torch.ones_like(flare)
        zero = torch.zeros_like(flare)
        flare = torch.where(flare < 0.1, zero, one)
        mask_loss = self.loss(source, flare)
        return self.loss_weight*mask_loss
@LOSS_REGISTRY.register()
class MaskBaseLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskBaseLoss, self).__init__()
        self.loss_weight=loss_weight
        self.loss = nn.SmoothL1Loss()
        self.to_gray = transforms.Grayscale(num_output_channels=1)
    def forward(self, output,target,flare):
        flare = self.to_gray(flare)
        one = torch.ones_like(flare)
        zero = torch.zeros_like(flare)
        flare = torch.where(flare < 0.1, zero, one)
        maskbase_loss = self.loss(output*flare, target*flare)
        return self.loss_weight*maskbase_loss
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

@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        label_fft = torch.fft.fft2(target, dim=(-2, -1))
        label_fft = torch.stack((label_fft.real, label_fft.imag), -1)

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft.real, pred_fft.imag), -1)

        loss=self.l1(pred_fft3,label_fft)*self.loss_weight

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
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):

    def __init__(self, loss_weight=1.0,):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ssim = MS_SSIMLoss()



    def forward(self, pred, target):

        loss=self.ssim(pred,target)*self.loss_weight
        return loss

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
# @LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1)
    https://arxiv.org/abs/1704.03915
    Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
    """

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

@LOSS_REGISTRY.register()
class SFLoss(nn.Module):
    def __init__(self):
        super(SFLoss, self).__init__()
        self.criterion = nn.L1Loss()
    def forward(self,pred_img,label_img):

        label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
        l1 = self.criterion(pred_img[0], label_img4)
        l2 = self.criterion(pred_img[1], label_img2)
        l3 = self.criterion(pred_img[2], label_img)
        loss_content = l1 + l2 + l3

        label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

        pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

        pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        f1 = self.criterion(pred_fft1, label_fft1)
        f2 = self.criterion(pred_fft2, label_fft2)
        f3 = self.criterion(pred_fft3, label_fft3)
        loss_fft = f1 + f2 + f3

        loss = loss_content + 0.1 * loss_fft
        return loss

# Exposure Control Loss ： 曝光控制损失
