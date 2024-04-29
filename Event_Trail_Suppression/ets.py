"""
@Time ： 2022/12/2 20:38
@Auth ： Haoyue Liu
@File ：ets.py
"""
import argparse
from tqdm import tqdm
from ets_utils import *


def ets_process(events, t0, s_w, s_h, threshold_t_on, threshold_t_off, soft_thr):
    # ----------------------------Grid the events according to coordinates, with each pixel containing a sequence of timestamp values.----------------------------
    # Create two empty lists with a shape of [H, W].
    ts_map = [[[] for _ in range(s_w)] for _ in range(s_h)]
    p_map = [[[] for _ in range(s_w)] for _ in range(s_h)]

    # Traverse the array events and append t(i) to the list at the corresponding position in X.
    for ev in tqdm(events):
        ts_, xs_, ys_, ps_ = ev[0], ev[1], ev[2], ev[3]
        ts_map[ys_][xs_].append(ts_)
        p_map[ys_][xs_].append(ps_)
    ts_map = np.array(ts_map)
    p_map = np.array(p_map)

    # Each element t_array in ts_map represents the timestamps of all events triggered at a pixel point (xx, yy). Convert the two-dimensional matrix into a one-dimensional array.
    ts_map = np.concatenate([np.array(row) for row in ts_map if len(row) > 0])
    p_map = np.concatenate([np.array(row) for row in p_map if len(row) > 0])

    # ----------------------------------------ETS processing----------------------------------------
    ets_events = np.ones((len(events), 4)) * -1
    n_evs = 0

    for ii, t_array in tqdm(enumerate(ts_map)):
        # Skip elements that are empty lists.
        if not t_array:
            continue
        xx = ii % s_w
        yy = int((ii - xx) / s_w)
        t_array = np.array(t_array)
        if len(np.atleast_1d(t_array)) == 1:
            p_array = np.array(p_map[ii])
            ets_events[n_evs] = np.array([t_array, xx, yy, p_array])
            n_evs += 1
        else:
            sort_id = np.argsort(t_array)
            t_array = t_array[sort_id]
            p_array = np.array(p_map[ii])[sort_id]

            for nn in range(len(t_array)):
                if nn == 0:
                    num = 0
                    previous_p = p_array[nn]
                    previous_t = t_array[nn]
                    start_t = previous_t
                    time_interval = 0
                else:
                    if p_array[nn] == 1:
                        threshold_t = threshold_t_on
                    else:
                        threshold_t = threshold_t_off
                    # Events triggered within the same polarity, where the time interval since the last event is greater than the previous interval but less than the threshold value threshold_t.
                    if p_array[nn] == previous_p and t_array[nn] - previous_t > time_interval and t_array[nn] - previous_t < threshold_t:
                        # For events that meet the tailing condition, modify their triggering timestamps to be the time of the previous event triggered at that pixel plus 1 microsecond.
                        # Update iteration parameters.
                        num += 1
                        time_interval = t_array[nn] - previous_t - soft_thr
                        previous_t = t_array[nn]
                        t_array[nn] = start_t + num  # Correct timestamps.
                        # start_t = previous_t
                        previous_p = p_array[nn]
                    else:
                        # If the condition is not met, initialize parameters and start the next iteration
                        num = 0
                        previous_p = p_array[nn]
                        previous_t = t_array[nn]
                        start_t = previous_t
                        time_interval = 0

                ets_events[n_evs] = np.array([t_array[nn], xx, yy, p_array[nn]])
                n_evs += 1

    ets_events = ets_events.reshape(-1, 4)
    ets_events[:, 0] = ets_events[:, 0] + t0
    # Reorder the events processed by ETS based on their timestamps
    idex = np.lexsort([ets_events[:, 0]])
    ets_events = ets_events[idex, :]

    # Release memory
    del ts_map, p_map

    return ets_events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Events trail suppression')
    parser.add_argument('--root', type=str, default='./data/test_20.0ms/h5', help='Root directory of the data.')
    parser.add_argument('--w', type=int, default=1280, help='Sensor width')
    parser.add_argument('--h', type=int, default=720, help='Sensor Height')

    parser.add_argument('--cut', action='store_true', help=' If true, cut file according to the given timestamps.')
    parser.add_argument('--start', type=int, default=0, help='Start timestamp(us)')
    parser.add_argument('--end', type=int, default=5000000, help='End timestamp(us)')

    parser.add_argument('--frame', action='store_true', help=' If true, save event frames')
    parser.add_argument('--interval', type=int, default=20000, help='Interval time for generating event frames(us)')

    parser.add_argument('--h5', action='store_true', help=' If true, save .h5 file')
    parser.add_argument('--npy', action='store_true', help=' If true, save .npy files')

    """ ETS parameters """
    parser.add_argument('--t_on', type=int, default=1e6, help='Maximal time range for executing the ETS method (ON events, us).')
    parser.add_argument('--t_off', type=int, default=1e6, help='Maximal time range for executing the ETS method (OFF events, us).')
    parser.add_argument('--soft_t', type=int, default=0, help='When assessing the length of two adjacent time intervals, add an allowable margin of error(us).')

    args = parser.parse_args()

    root = args.root
    if os.path.isdir(root):
        h5_file_list = get_h5_file(root)
    else:
        h5_file_list = [root]

    # ---------------------------------Loop through all H5 files for processing-----------------------------------------
    for data_id, h5_file in enumerate(h5_file_list):
        print(h5_file)
        if os.path.isdir(root):
            root_path = root
        else:
            root_path = os.path.dirname(root)
        frame_save_path = ''
        if args.frame:
            frame_save_path = os.path.join(root_path, 'trail_filter_result/event_frame')
            ensure_dir(frame_save_path)
        npy_save_path = ''
        if args.npy:
            npy_save_path = os.path.join(root_path, 'trail_filter_result/npy')
            ensure_dir(npy_save_path)
        h5_save_path = ''
        if args.h5:
            h5_save_path = os.path.join(root_path, 'trail_filter_result')
            ensure_dir(h5_save_path)

        x, y, t, p = read_h5_event_components(h5_file)
        # t = t/1e6  # Convert timestamps to us
        p[p == -1] = 0

        # Cut file according to the given timestamps.
        if args.cut:
            idx = np.where((t >= args.start) & (t <= args.end))[0]
            t = t[idx]
            x = x[idx]
            y = y[idx]
            p = p[idx]

        t_0 = t[0]
        t = t - t[0]
        evs = np.vstack((t, x, y, p)).transpose().astype(np.int64)  # Structure events in the format [t, x, y, p] (n * 4).

        # ETS processing procedure
        ets_events = ets_process(evs, t_0, args.w, args.h, args.t_on, args.t_off, args.soft_t)

        if args.h5:
            file_name = os.path.join(h5_save_path, os.path.basename(h5_file))
            save_h5_file_with_image(file_name, ets_events, args.w, args.h, h5_file)
        if args.npy:
            ets_npy_path = os.path.join(root_path, 'trail_filter_result/ets_events.npy')
            np.save(ets_npy_path, ets_events)

        t_, x_, y_, p_ = (ets_events[:, 0]).astype(np.int64), (ets_events[:, 1]).astype(np.uint16), \
            (ets_events[:, 2]).astype(np.uint16), (ets_events[:, 3]).astype(np.uint8)

        start_timestamp = t_[0]
        p_[p_ < 0] = 0  # "ON":1, "OFF":0
        t_ = (t_ - start_timestamp).astype(np.int64)

        if args.npy or args.frame:
            save_evs_frame_and_npy(args, frame_save_path, npy_save_path, ets_events, x_, y_, t_, p_, args.interval)
