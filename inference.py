import argparse
import torch
import os
from os.path import join, basename, dirname
import cv2
from tqdm import tqdm
import glob

from model.load_util import load_model
from data_loader.data_loaders import InferenceDataLoader_evs
from utils.util import CropParameters, get_height_width, get_height_width_evs, torch2cv2
from utils.timers import CudaTimer, Timer
from utils.util import ensure_dir
from data_loader.dataset import render


def set_parser(parser):
    parser.add_argument('--checkpoint_path', default='./pretrained', type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--model_name', default='ner.pth', type=str, help='checkpoint file name')
    parser.add_argument('--dataset_path', default="./data", type=str, help='path to datasets (HDF5)')
    parser.add_argument('--output_folder', default='./data/reconstruction_result', type=str, help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str, help='indices of GPUs to enable')
    parser.add_argument('--loader_type', default='H5', type=str, help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--save_gt', action='store_true', default=False, help='If true, save ground truth')
    parser.add_argument('--save_evframe', action='store_true', default=False, help='If true, save event frames')
    args = parser.parse_args()

    return args


def arch(model_path, n, device, sensor_resolution):
    model_path = join(model_path, n)
    netG = load_model(model_path, device, sensor_resolution)
    if torch.cuda.is_available():
        return netG.eval().cuda()
    else:
        return netG.eval()


def generate_output_dirs(args, root_folder, model_name, data_path):
    model_name_ = model_name.replace('.pth', '').replace('.tar', '')
    data_name = basename(data_path).split('/')[-1].split('.')[0]
    output_folder = join(join(root_folder, model_name_), data_name)
    ensure_dir(output_folder)
    if args.save_gt:
        args.gt_path = join(join(root_folder, model_name_), data_name + '_gt')
        ensure_dir(args.gt_path)
    if args.save_evframe:
        args.evframe_path = join(join(root_folder, model_name_), data_name + '_evs')
        ensure_dir(args.evframe_path)
    args.dataset_path = data_path
    args.output_folder = output_folder

    return args


@torch.no_grad()
def main(args, model_name, device):
    # set dataloader
    data_loader = InferenceDataLoader_evs(args.dataset_path, num_workers=0, dataset_kwargs={}, ltype=args.loader_type)
    height, width = get_height_width_evs(data_loader)
    sensor_resolution = (height, width)

    # load model
    model = arch(args.checkpoint_path, model_name, device, sensor_resolution).to(device)  # model_name, net_param
    model.reset_states()

    # set CUDA timer
    timer = CudaTimer if torch.cuda.is_available() else Timer

    for i, item in enumerate(tqdm(data_loader, mininterval=10)):
        events = item['events'].to(device)  # [1, B, H, W]

        with timer('Inference'):
            output, pred_voxel = model(events)

        # save reconstruction results
        fname_name = 'frame_{:05d}.png'.format(i)
        output_img = output['image']
        output_img = torch.squeeze(output_img)
        image = torch2cv2(output_img)  # tensor to [0, 255]
        cv2.imwrite(join(args.output_folder, fname_name), image)

        # save event frames
        if args.save_evframe:
            evs = item['evs'].cpu().numpy()
            frame = render(evs, height, width)
            cv2.imwrite(os.path.join(args.evframe_path, 'evs_{:05d}.png'.format(i)), frame)

        # save gt frames
        if args.save_gt:
            gt = item['frame']
            im_gt = torch2cv2(gt)
            cv2.imwrite(join(args.gt_path, fname_name), im_gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    args = set_parser(parser)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    data_path_list = []
    if os.path.isdir(args.dataset_path):
        data_path_list = sorted(glob.glob(os.path.join(args.dataset_path, '*.h5')))
    else:
        data_path_list.append(args.dataset_path)

    print(f'--------------{args.model_name} inferring------------')
    root_folder = args.output_folder
    for data_path in data_path_list:
        args = generate_output_dirs(args, root_folder, args.model_name, data_path)
        main(args, args.model_name, device)
