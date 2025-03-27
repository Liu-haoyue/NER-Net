"""
@Time ： 2023/5/20 11:48
@Auth ： Haoyue Liu
@File ：visualize_data.py
"""
import os
from os.path import join, exists, basename
import glob
from tqdm import tqdm
import numpy as np
import cv2

from data_loader.data_loaders import InferenceDataLoader
from utils.util import get_height_width


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')
    mask = np.zeros((H, W), dtype='int32')
    pol = pol.astype('int')
    pol[pol == 0] = -1
    mask1 = (x >= 0) & (y >= 0) & (W > x) & (H > y)
    mask[y[mask1], x[mask1]] = pol[mask1]
    img[mask == 0] = [255, 255, 255]
    img[mask == -1] = [255, 0, 0]
    img[mask == 1] = [0, 0, 255]
    return img


def get_data_visualize(dataset_path, output_path):

    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 5,
                      'filter_hot_events': False,
                      'voxel_method': {'method': 'between_frames'}
                      }
    data_path_list = []
    if os.path.isdir(dataset_path):
        data_path_list = sorted(glob.glob(join(dataset_path, '*.h5')))
    else:
        data_path_list.append(dataset_path)
    for events_file_path in data_path_list:
        event_path = join(output_path, basename(events_file_path).replace('.h5', ''), 'event_frames')
        image_path = join(output_path, basename(events_file_path).replace('.h5', ''), 'images')
        if not exists(event_path):
            os.makedirs(event_path)
        if not exists(image_path):
            os.makedirs(image_path)

        data_loader = InferenceDataLoader(events_file_path, num_workers=0, dataset_kwargs=dataset_kwargs, ltype='H5')
        height, width = get_height_width(data_loader)

        for i, item in enumerate(tqdm(data_loader, mininterval=10)):
            evs = item['evs'][0].cpu().numpy()  # n * 4([x, y, t, p)
            x = evs[:, 0].astype(np.int64)  # s
            y = evs[:, 1].astype(np.int64)
            t = evs[:, 2]
            p = evs[:, 3]
            evs_frame = render(x, y, p, height, width)
            save_evs =join(event_path, str(i).zfill(8) + '.png')
            cv2.imwrite(save_evs, evs_frame)

            image = item['raw_frame'][0].cpu().numpy()
            save_image = join(image_path, str(i).zfill(8) + '.png')
            cv2.imwrite(save_image, image)


if __name__ == '__main__':
    h5_file_path = 'data'
    save_path = 'data/visualize'
    get_data_visualize(h5_file_path, save_path)
