import numpy as np
import torch
# local modules
from torch.utils.data import DataLoader
from .dataset import MemMapDataset
from .dataset_evs import SequenceDataset_evs
from .dataset_evs_infer import DynamicH5Dataset_evs
from utils.data import concatenate_subfolders, concatenate_datasets


def collate_events_train(data):
    sequence = []
    # print('---collate_events---')
    for i in range(len(data[0])):
        frames = [d[i]['frame'] for d in data]
        frames = torch.stack(frames, dim=0)
        flows = [d[i]['flow'] for d in data]
        flows = torch.stack(flows, dim=0)
        events = [torch.cat((d[i]['events'], j*torch.ones((len(d[i]['events']), 1), dtype=torch.float32)), dim=1) for j, d in enumerate(data)]
        events = torch.cat(events, dim=0)
        timestamp = torch.tensor([d[i]['timestamp'] for d in data])
        data_source_idx = torch.tensor(([d[i]['data_source_idx'] for d in data]))
        dt = torch.tensor([d[i]['dt'] for d in data])

        # raw dataset: frame: [1, crop_h, crop_w]; flow: [2, crop_h, crop_w]; voxel: [5, crop_h, crop_w]
        item = {'frame': frames,
                'flow': flows,
                'events': events,
                'timestamp': timestamp,
                'data_source_idx': data_source_idx,
                'dt': dt
                }

        sequence.append(item)

    return sequence


class MemMapDataLoader_evs(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=0,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset_evs, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_events_train)


def collate_events(data):
    raw_events = data[0]['events']  # (n, 4), n * [t, x, y, p]
    events = []
    for i, d in enumerate(data):
        ev = np.concatenate([d['events'], i*np.ones((len(d['events']), 1), dtype=np.float32)], 1)
        events.append(ev)  # ev: (n, 5), [x, y, t, p, b(batch id)]
    events = torch.from_numpy(np.concatenate(events, 0)).float()
    if 'frame' in data[0]:
        item = {'has_frame': data[0]['has_frame'],
                'events': events,
                'frame': data[0]['frame'],
                'timestamp': data[0]['timestamp'],
                'dt': data[0]['dt'],
                'evs': raw_events}
    else:
        item = {'has_frame': data[0]['has_frame'],
                'events': events,
                'timestamp': data[0]['timestamp'],
                'dt': data[0]['dt'],
                'evs': raw_events}
    return item


class InferenceDataLoader_evs(DataLoader):
    def __init__(self, data_path, num_workers=1, pin_memory=False, dataset_kwargs=None, ltype="H5"):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if ltype == "H5":
            dataset = DynamicH5Dataset_evs(data_path, **dataset_kwargs)
        elif ltype == "MMP":
            dataset = MemMapDataset(data_path, **dataset_kwargs)
        else:
            raise Exception("Unknown loader type {}".format(ltype))
        super().__init__(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_events)
