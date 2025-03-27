import argparse
import h5py
import numpy as np
import cv2
import os
import glob
import json


def find_safe_alternative(output_base_path):
    i = 0
    alternative_path = "{}_{:09d}".format(output_base_path, i)
    while (os.path.exists(alternative_path)):
        i += 1
        alternative_path = "{}_{:09d}".format(output_base_path, i)
        assert (i < 999999999)
    return alternative_path


def save_additional_data_as_mmap(f, mmap_pth, data):
    data_path = os.path.join(mmap_pth, data['mmap_filename'])
    data_ts_path = os.path.join(mmap_pth, data['mmap_ts_filename'])
    data_event_idx_path = os.path.join(mmap_pth, data['mmap_event_idx_filename'])
    data_key = data['h5_key']
    print('Writing {} to mmap {}, timestamps to {}'.format(data_key, data_path, data_ts_path))
    h, w, c = 1, 1, 1
    if data_key in f.keys():
        num_data = len(f[data_key].keys())
        if num_data > 0:
            data_keys = list(f[data_key].keys())
            data_size = f[data_key][data_keys[0]].attrs['size']
            h, w = data_size[0], data_size[1]
            c = 1 if len(data_size) <= 2 else data_size[2]
    else:
        num_data = 1

    # mmp_event_indices = np.memmap(data_event_idx_path, dtype='uint16', mode='w+', shape=(num_data, 1))

    if data_key in f.keys():
        data = []
        data_timestamps = []
        # data_event_index = []
        for img_key in f[data_key].keys():
            img = f[data_key][img_key][:]
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(img)
            data_timestamps.append(f[data_key][img_key].attrs['timestamp'])
            # data_event_index.append(f[data_key][img_key].attrs['event_idx'])
        data_stack = np.expand_dims(np.stack(data), axis=3)
        # data_ts_stack = np.expand_dims(np.stack(data_timestamps), axis=1)
        data_ts_stack = np.stack(data_timestamps)

        np.save(data_path, data_stack)
        np.save(data_ts_path, data_ts_stack)
        # np.save(data_event_idx_path, data_ts_stack)

    if 'event_indices' in f.keys():
        event_indices = f['event_indices']
        np.save(data_event_idx_path, event_indices)


def write_metadata(f, metadata_path):
    metadata = {}
    for attr in f.attrs:
        val = f.attrs[attr]
        if attr == 'sensor_resolution':
            val = [int(f.attrs[attr][1]), int(f.attrs[attr][0])]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if type(val) == np.uint32 or type(val) == np.int64 or type(val) == np.int32:
            val = int(val)
        metadata[attr] = val

    with open(metadata_path, 'w') as js:
        json.dump(metadata, js)


def h5_to_memmap(h5_file_path, output_base_path):
    mmap_pth = os.path.join(output_base_path, os.path.basename(h5_file_path).replace('.hdf5', '').replace('.h5', ''))
    print(mmap_pth)
    if not os.path.exists(mmap_pth):
        os.makedirs(mmap_pth)

    ts_path = os.path.join(mmap_pth, 't.npy')
    xy_path = os.path.join(mmap_pth, 'xy.npy')
    ps_path = os.path.join(mmap_pth, 'p.npy')
    metadata_path = os.path.join(mmap_pth, 'metadata.json')

    additional_data = {
        "images":
            {
                'h5_key': 'images',
                'mmap_filename': 'images.npy',
                'mmap_ts_filename': 'timestamps.npy',
                'mmap_event_idx_filename': 'image_event_indices.npy',
                'dims': 3
            },
    }

    with h5py.File(h5_file_path, 'r') as f:
        mmp_ts = np.expand_dims(np.array(f['events/ts'][:], dtype='float64'), axis=1)
        mmp_xy = np.stack((f['events/xs'][:], f['events/ys'][:])).transpose().astype('int16')
        mmp_ps = np.expand_dims(np.array(f['events/ps'][:], dtype='uint8'), axis=1)

        np.save(ts_path, mmp_ts)
        np.save(xy_path, mmp_xy)
        np.save(ps_path, mmp_ps)

        for data in additional_data:
            save_additional_data_as_mmap(f, mmap_pth, additional_data[data])
        write_metadata(f, metadata_path)


def write_txt(save_dir, all_folders):
    train_txt = os.path.join(save_dir, 'train_memmap.txt')
    val_txt = os.path.join(save_dir, 'val_memmap.txt')
    train_path = []
    val_path = []
    for i, subfolder in enumerate(all_folders):
        path_txt = subfolder + '\n'
        # choose 10% data for validation
        if i % 10 == 0:
            val_path.append(path_txt)
        else:
            train_path.append(path_txt)

    with open(train_txt, 'w') as f:
        f.writelines(train_path)

    with open(val_txt, 'w') as f:
        f.writelines(val_path)


if __name__ == "__main__":
    RLED_data_path = r'D:\0\RLED'
    root_path = os.path.join(RLED_data_path, 'train')
    output_path = os.path.join(RLED_data_path, 'train_val_memmap')
    os.makedirs(output_path, exist_ok=True)

    h5_files = glob.glob(os.path.join(root_path, "**", "*.h5"), recursive=True)
    not_overwrite = False
    for path in h5_files:
        h5_to_memmap(path, output_path)

    # save training data path to .txt file
    subfolders = [os.path.join(root, d) for root, dirs, _ in os.walk(output_path) for d in dirs]
    write_txt(RLED_data_path, subfolders)
