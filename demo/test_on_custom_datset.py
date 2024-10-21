import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from natsort import os_sorted
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt


def calculate_psnr_ssim(sr, gt, crop_border, test_y):
    sr = sr.detach()
    gt = gt.detach()
    psnr = calculate_psnr_pt(sr, gt, crop_border, test_y)
    ssim = calculate_ssim_pt(sr, gt, crop_border, test_y)
    return psnr.item(), ssim.item()


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def load(path, model, key='state_dict', delete_module=False):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint if len(checkpoint) > 10 else checkpoint[key]

    model_dict = model.state_dict()
    if delete_module:
        checkpoint = delete_state_module(checkpoint)
    overlap = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(overlap)
    print(f'{(len(overlap) * 1.0 / len(checkpoint) * 100):.4f}% weights is loaded!', end='\t')
    print(f'{(len(overlap) * 1.0 / len(model_dict) * 100):.4f}% params is init!')
    print(f'Drop Keys: {[k for k, v in checkpoint.items() if k not in model_dict]}')
    model.load_state_dict(model_dict)
    return model


def delete_state_module(weights):
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    return weights_dict


def show_kv(path):
    ckpt = torch.load(path, map_location='cpu')['params_ema']
    for k, v in ckpt.items():
        print(f'{k}')


@torch.no_grad()
def test_on_custom_dataset(
        lr_path: str,
        hr_path: str,
        model,
        device,
        crop_border=0,
        test_y_channel=True,
        save=False,
        save_path='',
):
    if save:
        check_dir(save_path)

    lr_imgs = _get_paths_from_images(lr_path)
    hr_imgs = _get_paths_from_images(hr_path)

    model.to(device)
    model.eval()

    psnr, ssim = AverageMeter(), AverageMeter()

    for i, (lr_img, hr_img) in enumerate(zip(lr_imgs, hr_imgs)):
        base, ext = os.path.splitext(os.path.basename(lr_img))
        lr = cv2.imread(lr_img)[:, :, ::-1]
        hr = cv2.imread(hr_img)[:, :, ::-1]

        lr_tensor = uint2tensor(lr).to(device)
        hr_tensor = uint2tensor(hr).to(device)
        output = model(lr_tensor)

        psnr_temp, ssim_temp = calculate_psnr_ssim(output, hr_tensor, crop_border, test_y_channel)
        psnr.update(psnr_temp)
        ssim.update(ssim_temp)

        # print(f'Processing {i}: LR:{lr_img} | HR:{hr_img} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

        if save:
            output_copy = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path, f'{base}{ext}'), output_copy[:, :, ::-1])

    avg_psnr = psnr.avg
    avg_ssim = ssim.avg
    # print(f'Avg PSNR:{avg_psnr} | Avg SSIM: {avg_ssim}')

    return avg_psnr, avg_ssim


@torch.no_grad()
def test_demo(model, device, input_path, save_path, suffix=''):
    model.to(device)
    model.eval()

    if os.path.isdir(input_path):
        lr_imgs = _get_paths_from_images(input_path)
    else:
        lr_imgs = input_path if isinstance(input_path, (tuple, list)) else [input_path]

    for im in tqdm(lr_imgs):
        base, ext = os.path.splitext(os.path.basename(im))
        lr = uint2tensor(cv2.imread(im)[:, :, ::-1]).to(device)
        output = model(lr)
        sr = tensor2uint8(output)
        cv2.imwrite(os.path.join(save_path, f'{base}{suffix}{ext}'), sr[:, :, ::-1])


def _model_dict_(scale, model_name, weight_path, key='state_dict', delete_module=False):
    from basicsr.archs.EIMN_arch import EIMN_A, EIMN_L

    if model_name == 'EIMN_A':
        model = EIMN_A(scale=scale)
    elif model_name == 'EIMN_L':
        model = EIMN_L(scale=scale)
    elif model_name == 'EIMN_beauty':
        from basicsr.archs.EIMN_baidu_arch import EIMNBaidu
        model = EIMNBaidu(scale=scale)
    load(weight_path, model, key=key, delete_module=delete_module)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scale = 2
    model_name = 'EIMN_beauty'  #'EIMN_L'
    # weight = r'./weights/Net/net_g_941000.pth'
    weight = r'./weights/Bicubic/x2_EIMN_Bicubic_DIV2K_enlarge5.pth'
    # weight = r'../experiments/pretrained_models/EIMN_L_x2.pth'
    model = _model_dict_(scale, model_name, weight, key='params', delete_module=False)

    lrs = [
        rf'E:\Dataset\Restoration\SR\Benchmark\Set5\LRbicx{scale}',
        rf'E:\Dataset\Restoration\SR\Benchmark\Set14\LRbicx{scale}',
        rf'E:\Dataset\Restoration\SR\Benchmark\Urban100\LRbicx{scale}',
        rf'E:\Dataset\Restoration\SR\Benchmark\Manga109\LRbicx{scale}',
        rf'E:\Dataset\Restoration\SR\Benchmark\BSDS100\LRbicx{scale}',
    ]
    hrs = [
        rf'E:\Dataset\Restoration\SR\Benchmark\Set5\GTmod12',
        rf'E:\Dataset\Restoration\SR\Benchmark\Set14\GTmod12',
        rf'E:\Dataset\Restoration\SR\Benchmark\Urban100\GTmod12',
        rf'E:\Dataset\Restoration\SR\Benchmark\Manga109\GTmod12',
        rf'E:\Dataset\Restoration\SR\Benchmark\BSDS100\GTmod12',
    ]

    for lr_p, hr_p in zip(lrs, hrs):
        name = hr_p.split(os.sep)[-2]
        psnr, ssim = test_on_custom_dataset(
            # Custom LR/HR images dir path
            lr_path=lr_p,
            hr_path=hr_p,
            # Selected model
            model=model,
            device=device,
            # Test PSNR/SSIM configs
            crop_border=scale,
            test_y_channel=True,
            # Save output or not
            save=False,
            save_path=r''
        )

        print(f'{name} {psnr:.4f}/{ssim:.4f}')

    # test_demo(
    #     model=model,
    #     device=device,
    #     # input_path=r'G:\PhD\Paper\INFFUS\Codes\datasets\DIV2K\DIV2K_valid_LR_difficult\0823x4d.png',
    #     # input_path=r'E:\Dataset\Restoration\SR\Benchmark\Set5\LRbicx2\baby.png',
    #     input_path=r'./input',
    #     save_path=r'./output',
    #     # suffix='_BicWeight'
    #     suffix='_328000_GANX4'
    # )

"""
DIV2K_mulitiscale_p480:
Set5 38.2293/0.9619
Set14 33.9597/0.9212
Urban100 32.9755/0.9367
Manga109 39.3294/0.9783
BSDS100 32.3638/0.9029

DIV2K_enlarge_5:
Set5 38.2199/0.9619
Set14 33.9294/0.9218
Urban100 32.8558/0.9354
Manga109 39.2114/0.9776
BSDS100 32.3419/0.9027
"""
"""
ToDO List

1. Degradation pipeline X2/4 Net/GAN
2. Paired human face beatiful  qudou
3. meiyan + degradation pipeline


"""