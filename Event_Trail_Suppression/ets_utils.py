import os
from abc import ABCMeta, abstractmethod
import h5py
import numpy as np
import cv2


class packager():
    __metaclass__ = ABCMeta

    def __init__(self, name, output_path, max_buffer_size=1000000):
        self.name = name
        self.output_path = output_path
        self.max_buffer_size = max_buffer_size

    @abstractmethod
    def package_events(self, xs, ys, ts, ps):
        pass

    @abstractmethod
    def package_image(self, frame, timestamp):
        pass

    @abstractmethod
    def package_flow(self, flow, timestamp):
        pass

    @abstractmethod
    def add_metadata(self, num_events, num_pos, num_neg,
                     duration, t0, tk, num_imgs, num_flow):
        pass

    @abstractmethod
    def set_data_available(self, num_images, num_flow):
        pass


class hdf5_packager(packager):
    """
    This class packages data to hdf5 files
    """

    def __init__(self, output_path, max_buffer_size=1000000):
        packager.__init__(self, 'hdf5', output_path, max_buffer_size)
        print("CREATING FILE IN {}".format(output_path))
        self.events_file = h5py.File(output_path, 'w')
        self.event_xs = self.events_file.create_dataset("events/xs", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True)
        self.event_ys = self.events_file.create_dataset("events/ys", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True)
        self.event_ts = self.events_file.create_dataset("events/ts", (0,), dtype=np.dtype(np.float64), maxshape=(None,), chunks=True)
        self.event_ps = self.events_file.create_dataset("events/ps", (0,), dtype=np.dtype(np.bool_), maxshape=(None,), chunks=True)

    def append_to_dataset(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data):] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append_to_dataset(self.event_xs, xs)
        self.append_to_dataset(self.event_ys, ys)
        self.append_to_dataset(self.event_ts, ts)
        self.append_to_dataset(self.event_ps, ps)

    def package_image(self, image, timestamp, img_idx):
        image_dset = self.events_file.create_dataset("images/image{:09d}".format(img_idx),
                                                     data=image, dtype=np.dtype(np.uint8))
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr"

    def package_flow(self, flow_image, timestamp, flow_idx):
        flow_dset = self.events_file.create_dataset("flow/flow{:09d}".format(flow_idx),
                                                    data=flow_image, dtype=np.dtype(np.float32))
        flow_dset.attrs['size'] = flow_image.shape
        flow_dset.attrs['timestamp'] = timestamp

    def add_event_indices(self):
        datatypes = ['images', 'flow']
        for datatype in datatypes:
            if datatype in self.events_file.keys():
                s = 0
                added = 0
                ts = self.events_file["events/ts"][s:]
                for image in self.events_file[datatype]:
                    img_ts = self.events_file[datatype][image].attrs['timestamp']
                    event_idx = np.searchsorted(ts, img_ts)
                    if event_idx == len(ts):
                        added += len(ts)
                        s += self.max_buffer_size
                        ts = self.events_file["events/ts"][s:s + self.max_buffer_size]
                        event_idx = np.searchsorted(ts, img_ts)
                    event_idx = max(0, event_idx - 1)
                    self.events_file[datatype][image].attrs['event_idx'] = event_idx + added

    def add_metadata(self, num_pos, num_neg,
                     duration, t0, tk, num_imgs, num_flow, sensor_size):
        # duration, t0, tk, length, num_imgs, num_flow, sensor_size):
        self.events_file.attrs['num_events'] = num_pos + num_neg
        self.events_file.attrs['num_pos'] = num_pos
        self.events_file.attrs['num_neg'] = num_neg
        self.events_file.attrs['duration'] = tk - t0
        self.events_file.attrs['t0'] = t0
        self.events_file.attrs['tk'] = tk
        self.events_file.attrs['num_imgs'] = num_imgs
        self.events_file.attrs['num_flow'] = num_flow
        self.events_file.attrs['sensor_resolution'] = sensor_size

    def add_indices(self, indices_list):
        self.events_file.create_dataset("event_indices", data=indices_list)

    def set_data_available(self, num_images, num_flow):
        if num_images > 0:
            self.image_dset = self.events_file.create_group("images")
            self.image_dset.attrs['num_images'] = num_images
        if num_flow > 0:
            self.flow_dset = self.events_file.create_group("flow")
            self.flow_dset.attrs['num_images'] = num_flow


def read_h5_event_components(hdf_path):
    """
    Read events from HDF5 file. Return x,y,t,p components.
    """
    with h5py.File(hdf_path, 'r') as f:
        if 'events/x' in f:
            f_x, f_y, f_t, f_p = f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1)
        else:
            f_x, f_y, f_t, f_p = f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1)
    return f_x, f_y, f_t, f_p


def save_evs_frame_and_npy(args, frame_save_path, npy_save_path, ets_events, x_save, y_save, t_save, p_save, delta_t):
    frame_id = 0
    start_idx = 0
    last_stamp = t_save[0]

    for i, ts in enumerate(t_save):
        if ts > last_stamp + delta_t:
            if args.npy:
                slice_events = ets_events[start_idx: i]
                npy_name = str(frame_id).zfill(5) + '.npy'
                np.save(os.path.join(npy_save_path, npy_name), slice_events)
            if args.frame:
                # save event frames
                x1 = x_save[start_idx: i]
                y1 = y_save[start_idx: i]
                p1 = p_save[start_idx: i]
                event_frame = render(x1, y1, p1, args.h, args.w)
                name = str(frame_id).zfill(5)
                cv2.imwrite(os.path.join(frame_save_path, name + '.png'), event_frame)
            start_idx = i
            last_stamp = ts
            frame_id += 1
        # Process the remaining events in the last iteration."
        if i == len(t_save) - 1 and start_idx < i:
            if args.npy:
                slice_events = ets_events[start_idx: i]
                npy_name = str(frame_id).zfill(5) + '.npy'
                np.save(os.path.join(npy_save_path, npy_name), slice_events)
            if args.frame:
                # save event frames
                x1 = x_save[start_idx:]
                y1 = y_save[start_idx:]
                p1 = p_save[start_idx:]
                event_frame = render(x1, y1, p1, args.h, args.w)
                name = str(frame_id).zfill(5)
                cv2.imwrite(os.path.join(frame_save_path, name + '.png'), event_frame)


def save_h5_file_with_image(h5_path, events, w, h, source_file):
    sensor_size = (h, w)
    h5_save_file = h5_path
    ep = hdf5_packager(h5_save_file)
    t_h5, x_h5, y_h5, p_h5 = (events[:, 0]).astype(np.int64), (events[:, 1]).astype(np.uint16), (events[:, 2]).astype(np.uint16), (events[:, 3]).astype(np.uint8)
    t0 = t_h5[0]
    p_h5[p_h5 < 0] = 0  # should be [0 or 1]
    ep.set_data_available(num_images=0, num_flow=0)
    p_h5 = p_h5.astype(bool)
    last_ts = t_h5[-1]
    total_num_pos = len(p_h5[p_h5 == 1])
    total_num_neg = len(p_h5[p_h5 == 0])
    ep.package_events(x_h5, y_h5, t_h5, p_h5)
    ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)

    # Copy images to the new HTML5 file.
    with h5py.File(source_file, 'r') as source:
        if 'images' in source.keys():
            timestamps_list = []
            images_list = []
            for im_id, img_key in enumerate(source['images'].keys()):
                timestamps = source['images'][img_key].attrs['timestamp']
                timestamps_list.append(timestamps)
                image = np.array(source['images'][img_key])
                images_list.append(image)
                ep.package_image(image, timestamps, im_id)
            ep.add_event_indices()


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


def ensure_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def get_h5_file(data_path):
    '''
    find imgs files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['h5', 'hdf5']
    isroot = True
    for parent, dirnames, filenames in os.walk(data_path):
        if isroot:
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        isroot = False
    print(('Find {} h5 files'.format(len(files))))
    files.sort()
    return files
