from collections import OrderedDict
from os import path as osp

from click.core import F
import torch.nn.functional as F1
# from Demos.SystemParametersInfo import x
# from click.core import F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.flare_util import blend_light_source, mkdir, predict_flare_from_6_channel, predict_flare_from_3_channel
from kornia.metrics import psnr, ssim
from basicsr.metrics import calculate_metric
import torch
from tqdm import tqdm
from torch import nn


@MODEL_REGISTRY.register()
class DeflareModel(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.output_ch = self.opt['network_g']['output_ch']
        if 'multi_stage' in self.opt['network_g']:
            self.multi_stage = self.opt['network_g']['multi_stage']
        else:
            self.multi_stage = 1
        print("Output channel is:", self.output_ch)
        print("Network contains", self.multi_stage, "stages.")

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        # self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
        self.l1_pix = nn.SmoothL1Loss(reduction='mean').to(self.device)
        self.l_perceptual = build_loss(train_opt['perceptual']).to(self.device)
        self.mse = nn.MSELoss(reduction='mean').to(self.device)
        self.fft = build_loss(train_opt['FFT']).to(self.device)
        # self.maskbase = build_loss(train_opt['flarebase']).to(self.device)
        # self.ssim = build_loss(train_opt['ssim']).to(self.device)
        # self.psnr = build_loss(train_opt['psnr']).to(self.device)
        # self.HF = build_loss(train_opt['HFLoss']).to(self.device)
        # self.bce = nn.BCELoss(reduction='mean').to(self.device)
        # self.focal_freq_loss = build_loss(train_opt['FocalFrequencyLoss']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.metric_results = {'psnr': 0, 'ssim': 0}

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'flare' in data:
            self.flare = data['flare'].to(self.device)
            self.gamma = data['gamma'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        if self.output_ch == 6:
            self.deflare, self.flare_hat = predict_flare_from_6_channel(self.output)
        elif self.output_ch == 3:
            self.deflare = self.output
            # self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            # self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.lq,self.flare,self.lq,self.gamma)
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        l_total = 0
        loss_dict = OrderedDict()
        # l1 loss
        if self.output_ch == 6:
            l1_flare = self.l1_pix(self.flare_hat, self.flare)
            l1_base = self.l1_pix(self.deflare, self.gt)
            l1 = l1_flare + l1_base

            #     l1_recons= self.l1_pix(self.merge_hat, self.lq)
            #     loss_dict['l1_recons']=l1_recons*2
            #     l1+=l1_recons*2
            l_total += l1
            loss_dict['l1_flare'] = l1_flare
            loss_dict['l1_base'] = l1_base
            loss_dict['l1'] = l1

            # 基础掩码损失
            # maskbase = self.maskbase(self.deflare, self.gt, self.flare)
            # l_total += maskbase
            # loss_dict['maskbase'] = maskbase
            # 二元交叉熵损失
            # bce = self.bce(self.flare_hat, self.flare)
            # l_total += bce
            # loss_dict['bce'] = bce

            # perceptual loss
            l_vgg_flare = self.l_perceptual(self.flare_hat, self.flare)
            l_vgg_base = self.l_perceptual(self.deflare, self.gt)
            l_vgg = l_vgg_base + l_vgg_flare
            l_total += l_vgg
            loss_dict['l_vgg'] = l_vgg
            loss_dict['l_vgg_base'] = l_vgg_base
            loss_dict['l_vgg_flare'] = l_vgg_flare

        else:
            # self.gt1 = F1.interpolate(self.gt, scale_factor=0.25,  mode='bilinear')  #The size of tensor a (512) must match the size of tensor b (128) at non-singleton dimension
            # self.gt2 = F1.interpolate(self.gt, scale_factor=0.5, mode='bilinear')
            # self.gt = F1.interpolate(self.gt,scale_factor=0.25,mode='bilinear')
            # self.gt = F1.interpolate(self.gt, scale_factor=0.5, mode='bilinear')

            l1 = self.l1_pix(self.deflare, self.gt)    # AdaIR,MFDNet,Restormer
            # l1_1 = self.l1_pix(self.deflare[0], self.gt1)  # ECFNet
            # l1_2 = self.l1_pix(self.deflare[1], self.gt2)
            # l1_3 = self.l1_pix(self.deflare[2], self.gt)
            # l1_1 = self.l1_pix(self.deflare[0], self.gt)
            # l1_2 = self.l1_pix(self.deflare[1], self.gt)
            # l1_3 = self.l1_pix(self.deflare[2], self.gt)
            # l1=l1_1+l1_2+l1_3
            # l1 = l1_1 + l1_2
            l_total += l1
            loss_dict['l1'] = l1

            # perceptual loss
            l_vgg = self.l_perceptual(self.deflare, self.gt)
            # l_vgg_1 = self.l_perceptual(self.deflare[0], self.gt1)
            # l_vgg_2 = self.l_perceptual(self.deflare[1], self.gt2)
            # l_vgg_3 = self.l_perceptual(self.deflare[2], self.gt)
            # l_vgg_1 = self.l_perceptual(self.deflare[0], self.gt)
            # l_vgg_2 = self.l_perceptual(self.deflare[1], self.gt)
            # l_vgg_3 = self.l_perceptual(self.deflare[2], self.gt)
            # l_vgg=l_vgg_1+l_vgg_2+l_vgg_3
            # l_vgg = l_vgg_1 + l_vgg_2
            l_total += l_vgg
            loss_dict['l_vgg'] = l_vgg

            mse = self.mse(self.deflare, self.gt)
            l_total += mse
            loss_dict['mse'] = mse
            #
            # bce = self.bce(self.flare_hat,self.deflare)
            # l_total += bce
            # loss_dict['bce'] = bce

            # focal_freq_loss = self.focal_freq_loss(self.deflare, self.gt)
            # l_total += focal_freq_loss
            # loss_dict['FocalFrequencyLoss'] = focal_freq_loss

            # sf = self.sf(self.deflare, self.gt)
            # l_total += sf
            # loss_dict['sf'] = sf
            # cb = self.cb(self.deflare, self.gt)
            # l_total += cb
            # loss_dict['cb'] = cb

            # ssim = self.ssim(self.deflare, self.gt)
            # l_total += ssim
            # loss_dict['ssim'] = ssim

            # # #
            # maskbase = self.maskbase(self.deflare, self.gt, self.flare)
            # l_total += maskbase
            # loss_dict['maskbase']=maskbase
            #
            fft = self.fft(self.deflare, self.gt)
            l_total += fft
            loss_dict['fft'] = fft
            #
            # psnr = self.psnr(self.deflare, self.gt)
            # l_total += psnr
            # loss_dict['psnr'] = psnr
            #
            # HF = self.HF(self.deflare, self.gt)
            # l_total += HF
            # loss_dict['HF'] = HF

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.output_ch = self.opt['network_g']['output_ch']
        if self.output_ch == 6:
            self.deflare, self.flare_hat = predict_flare_from_6_channel(self.output)
        elif self.output_ch == 3:
            # self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            # self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.gt,self.flare,self.lq,self.gamma)
            self.deflare = self.output
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        total = len(dataloader)
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            img_name = 'deflare_' + str(idx).zfill(5) + '_'
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    # if self.opt['val']['suffix']:
                    #     save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                    #                              f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    # else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= total
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_metric_results(self):

        return self.metric_results['psnr']

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # self.blend= blend_light_source(self.lq, self.deflare, 0.97)
        out_dict['result'] = self.deflare.detach().cpu()  # MFDNet,Restormer
        # out_dict['result'] = self.deflare[2].detach().cpu()  # FocalNet,SANet,DSANet
        # out_dict['result'] = self.deflare[1].detach().cpu()  # HINet
        # out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
