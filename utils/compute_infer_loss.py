import torch
from model.loss import *
from model.model_util import mean

#from skimage.measure import compare_ssim as SSIM
#from PerceptualSimilarity.models import compare_ssim as SSIM
# from niqe import niqe

from joblib import load

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
# https://github.com/buyizhiyou/NRVQA
from utils.NRVQA.niqe import niqe
from utils.NRVQA.piqe import piqe
from utils.NRVQA.brisque import brisque
import numpy as np
import cv2


def LOE(I_raw, Ie_raw, keep_ratio_side=50):
    H_raw, W_raw = I_raw.shape

    r = keep_ratio_side / min(H_raw, W_raw)
    H = int(np.round(H_raw * r))
    W = int(np.round(W_raw * r))

    I = cv2.resize(I_raw, (W, H))
    Ie = cv2.resize(Ie_raw, (W, H))

    RD = np.zeros_like(I)
    for h in range(H):
        for w in range(W):
            RD[h, w] = np.sum(np.logical_xor(I[h, w] >= I, Ie[h, w] >= Ie))
    return (np.sum(RD) / (H * W))


class compute_infer_loss:
    def __init__(self, infer_loss_fname, LPIPS_net_type='vgg'):
        self.infer_loss_fname = infer_loss_fname
        self.perceptual_loss_fn = perceptual_loss(net=LPIPS_net_type, use_gpu=True)
        self.mse_loss_fn = l2_loss()
        self.ssim_loss_fn = SSIM
        self.loe_loss_fn = LOE
        self.niqe_loss_fn = niqe
        self.piqe_loss_fn = piqe
        self.loss = {'perceptual_loss': [],
                     'mse_loss': [],
                     'ssim_loss': [],
                     'loe_loss': [],
                     'niqe_loss': [],
                     'piqe_loss': []}

    def __call__(self, pred_img, gt_img):
        self.loss['perceptual_loss'].append(self.perceptual_loss_fn(pred_img, gt_img).item())
        self.loss['mse_loss'].append(self.mse_loss_fn(pred_img, gt_img).item())
        pred_cpu = pred_img.squeeze().cpu().numpy()
        gt_cpu = gt_img.squeeze().cpu().numpy()
        self.loss['ssim_loss'].append(self.ssim_loss_fn(pred_cpu,gt_cpu))
        self.loss['loe_loss'].append(self.loe_loss_fn(gt_cpu, pred_cpu))
        # self.loss['niqe_loss'].append(self.niqe_loss_fn(pred_cpu))
        # self.loss['piqe_loss'].append(self.piqe_loss_fn(pred_cpu)[0])
        # self.loss['niqe_loss'].append(pred_img.squeeze().cpu().numpy())

    def write_loss(self):
        mean_lpips = mean(self.loss['perceptual_loss'])
        mean_mse = mean(self.loss['mse_loss'])
        mean_ssim = mean(self.loss['ssim_loss'])
        mean_loe = mean(self.loss['loe_loss'])
        # mean_niqe = mean(self.loss['niqe_loss'])
        # mean_piqe = mean(self.loss['piqe_loss'])

        with open(self.infer_loss_fname, 'w') as f:
            f.write('perceptual loss for each step:{}\n'.format(self.loss['perceptual_loss']))
            f.write('mse loss for each step:{}\n'.format(self.loss['mse_loss']))
            f.write('ssim loss for each step:{}\n'.format(self.loss['ssim_loss']))
            f.write('******************************\n')
            f.write('mean mse loss for whole sequences:{}\n'.format(mean_mse))
            f.write('mean ssim loss for whole sequences:{}\n'.format(mean_ssim))
            f.write('mean perceptual loss for whole sequences:{}\n'.format(mean_lpips))
            f.write('mean loe loss for whole sequences:{}\n'.format(mean_loe))
            # f.write('mean niqe loss for whole sequences:{}\n'.format(mean_niqe))
            # f.write('mean piqe loss for whole sequences:{}\n'.format(mean_piqe))

        # return {'mean_lpips': mean_lpips, 'mean_mse': mean_mse, 'mean_ssim': mean_ssim, 'mean_niqe': mean_niqe, 'mean_piqe': mean_piqe}
        return {'mean_mse': mean_mse, 'mean_ssim': mean_ssim, 'mean_lpips': mean_lpips, 'mean_loe': mean_loe}


class compute_infer_loss_rescale:
    def __init__(self, infer_loss_fname, LPIPS_net_type='vgg'):
        self.infer_loss_fname = infer_loss_fname
        self.perceptual_loss_fn = perceptual_loss(net=LPIPS_net_type, use_gpu=True)
        self.mse_loss_fn = MSE
        self.ssim_loss_fn = SSIM
        self.niqe_loss_fn = niqe
        self.piqe_loss_fn = piqe
        self.brisque_loss_fn = brisque
        self.loss = {'perceptual_loss': [],
                     'mse_loss': [],
                     'ssim_loss': [],
                     'niqe_loss': [],
                     'piqe_loss': []}
        self.clf = load('NRVQA/svr_brisque.joblib')

    def rescale(self, image):
        image = torch.squeeze(image)  # H x W
        image = image.cpu().numpy()
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def __call__(self, pred_img, gt_img):
        self.loss['perceptual_loss'].append(self.perceptual_loss_fn(pred_img, gt_img).item())
        rescale_pred = self.rescale(pred_img)
        rescale_gt = self.rescale(gt_img)

        self.loss['mse_loss'].append(self.mse_loss_fn(rescale_pred, rescale_gt))
        self.loss['ssim_loss'].append(self.ssim_loss_fn(rescale_pred, rescale_gt))
        self.loss['niqe_loss'].append(self.niqe_loss_fn(rescale_pred))
        self.loss['piqe_loss'].append(self.piqe_loss_fn(rescale_pred)[0])
        # need scikit-learn==0.22
        # brisque_feat = self.brisque_loss_fn(rescale_pred).reshape(1, -1)
        # brisque_score = self.clf.predict(brisque_feat)[0]
        # print(brisque_score)
        # self.loss['brisque_loss'].append(brisque_score)

    def write_loss(self):
        mean_lpips = mean(self.loss['perceptual_loss'])
        mean_mse = mean(self.loss['mse_loss'])
        mean_ssim = mean(self.loss['ssim_loss'])
        mean_niqe = mean(self.loss['niqe_loss'])
        mean_piqe = mean(self.loss['piqe_loss'])
        # mean_brisque = mean(self.loss['brisque_loss'])

        with open(self.infer_loss_fname, 'w') as f:
            f.write('perceptual loss for each step:{}\n'.format(self.loss['perceptual_loss']))
            f.write('mse loss for each step:{}\n'.format(self.loss['mse_loss']))
            f.write('ssim loss for each step:{}\n'.format(self.loss['ssim_loss']))
            f.write('******************************\n')
            f.write('mean perceptual loss for whole sequences:{}\n'.format(mean_lpips))
            f.write('mean mse loss for whole sequences:{}\n'.format(mean_mse))
            f.write('mean ssim loss for whole sequences:{}\n'.format(mean_ssim))
            f.write('mean niqe loss for whole sequences:{}\n'.format(mean_niqe))
            f.write('mean piqe loss for whole sequences:{}\n'.format(mean_piqe))
            # f.write('mean brisque loss for whole sequences:{}\n'.format(mean_brisque))

        return {'mean_lpips': mean_lpips, 'mean_mse': mean_mse, 'mean_ssim': mean_ssim, 'mean_niqe': mean_niqe, 'mean_piqe': mean_piqe}
