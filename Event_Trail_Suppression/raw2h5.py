"""
@Time ： 2022/8/12 22:24
@Auth ： Haoyue Liu
@File ：read_format_events_PROPHESEE.py
@doc ： https://docs.prophesee.ai/stable/metavision_sdk/modules/core/tutorials/raw_dat_loading.html
        https://docs.prophesee.ai/stable/metavision_sdk/modules/core/tutorials/metavision_sdk_core_bindings.html
"""
from tqdm import tqdm

from ets_utils import *

""" Warning: Before using this code, please install the MetaVision SDK! """
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator


def save_events_h5(raw_path, data_iterator, h, w, file, save_im, save_np, save_h5file):
    frame_path = os.path.join(os.path.dirname(raw_path), os.path.basename(raw_path).replace('.raw', '') + '_' + file)
    frame_save_path = os.path.join(frame_path, 'event_frame')
    np_path = os.path.join(frame_path, 'npy')
    h5_path = os.path.join(frame_path, 'h5')
    if save_im:
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
    if save_np:
        if not os.path.exists(np_path):
            os.makedirs(np_path)
    if save_h5file:
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)
        h5_file = os.path.join(h5_path, os.path.basename(raw_path).replace('.raw', '.h5'))
        ep = hdf5_packager(h5_file)
        first_ts = -1
        t0 = -1
        sensor_size = [h, w]
        # Extract events to h5
        ep.set_data_available(num_images=0, num_flow=0)
        total_num_pos, total_num_neg, last_ts = 0, 0, 0
        slice_num = 0
        event_index = 0
        event_indices_list = []

    for frame_id, evs in enumerate(tqdm(data_iterator)):
        counter = evs.size
        if counter == 0:
            continue
        t, x, y, p = evs['t'], evs['x'], evs['y'], evs['p']
        counter = len(t)
        name = str(frame_id).zfill(5)
        if save_np:
            np.save(os.path.join(np_path, name + '.npy'), np.array([x, y, t, p], dtype=np.int64))
        if save_h5file:
            p[p < 0] = 0  # should be [0 or 1]
            p = p.astype(bool)
            if first_ts == -1:
                first_ts = t[0]
            last_ts = t[-1]
            sum_ps = sum(p)
            total_num_pos += sum_ps
            total_num_neg += len(p) - sum_ps
            ep.package_events(x, y, t, p)
            tmp_index = event_index
            event_index += counter
            event_indices_list.append([tmp_index, event_index])
            slice_num += 1

        # save event frames
        if save_im:
            event_frame = render(x, y, p, h, w)
            name = str(frame_id).zfill(5)
            cv2.imwrite(os.path.join(frame_save_path, name + '.png'), event_frame)

    if save_h5file:
        t0 = first_ts
        ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)
        # Release memory
        del ep


def read_file_prophesee(path, slice_time, dt, n, s_img, s_npy, s_h5):
    # open a file
    record_raw = RawReader(path)
    # print(record_raw)
    height, width = record_raw.get_size()
    # number of events to generate a frame
    if slice_time:
        file_name = str(dt/1000) + 'ms'
        mv_iterator = EventsIterator(input_path=path, mode='delta_t', delta_t=dt)
    else:
        file_name = str(n)
        mv_iterator = EventsIterator(input_path=path, mode='n_events', n_events=n)

    save_events_h5(path,  mv_iterator, height, width, file_name, s_img, s_npy, s_h5)


if __name__ == '__main__':
    file_path = 'data/test.raw'
    slice_by_time = True  # If TRUE, split events into fixed time intervals; otherwise, split events into fixed numbers of intervals.
    delta_t = 20000  # Time interval for each segment of events(us). (slice_by_time = True)
    n_events = 1000000  # Number of events per segment. (slice_by_time = False)
    save_img = True
    save_npy = True
    save_h5 = True

    read_file_prophesee(file_path, slice_by_time, delta_t, n_events, save_img, save_npy, save_h5)
