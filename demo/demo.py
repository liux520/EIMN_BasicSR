import torch
import os
from PIL import Image
import cv2
from tqdm import tqdm
import random
import numpy as np
from natsort import os_sorted


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


if __name__ == '__main__':
    scale = 2

    # save_path = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), 'outputs')
    # check_dir(save_path)

    # ----------- device ---------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------- model ----------- #
    from basicsr.archs.EIMN_baidu_arch import EIMNBaidu

    model = EIMNBaidu(scale=scale)

    load(r'./weights/beauty/net_g_latest.pth',
         model, key='params', delete_module=False)
    model.to(device)
    model.eval()

    # input path: directory or single-image path
    input_path = r'E:\Dataset\Misc\Baidu_beauty\train_datasets\image'
    save_path = r'G:\PhD\Paper\ECAI\Codes\EIMN_BasicSR\demo\EIMN_beauty_sr'
    os.makedirs(save_path, exist_ok=True)

    if os.path.isdir(input_path):
        lr_imgs = _get_paths_from_images(input_path)[100:]
    else:
        lr_imgs = [input_path]

    with torch.no_grad():
        for im in tqdm(lr_imgs):
            base, ext = os.path.splitext(os.path.basename(im))
            lr_np = cv2.imread(im)[:, :, ::-1]
            lr = uint2tensor(lr_np).to(device)
            output = model(lr)
            sr = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path, f'{base}{ext}'), sr[:, :, ::-1])

            h, w, c = sr.shape
            white = np.full((h, 1, 3), 255, dtype=np.uint8)
            lr_sr = np.concatenate((lr_np, white, sr), axis=1)
            cv2.imwrite(os.path.join(save_path, f'{base}_compare{ext}.png'), lr_sr[:, :, ::-1])
