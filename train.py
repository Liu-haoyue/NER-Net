import argparse
import collections
from collections import OrderedDict
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
from utils.util import get_height_width_evs
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, args):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, None)
    # logger.info(model)

    if args.representation_model:
        mapped_state_dict = {}
        if args.frozen_representation:
            # frozen letc module
            param_names = [name for name, _ in model.named_parameters()]
            frozen_layers = [name for name in param_names if 'representation' in name]
            for name, param in model.named_parameters():
                if any(frozen_layer in name for frozen_layer in frozen_layers):
                    print('forzen layer: ', name)  # e-raft = 124
                    param.requires_grad = False
        saved_weights = torch.load(args.representation_model, map_location='cpu' if torch.cuda.is_available else 'gpu')
        for name, param in saved_weights['state_dict'].items():
            mapped_name = "representation." + name
            print('load pretrained parameters: ', mapped_name)
            mapped_state_dict[mapped_name] = param
        # load parameters
        model.load_state_dict(OrderedDict(mapped_state_dict), strict=False)

    # init loss classes
    loss_ftns = [getattr(module_loss, loss)(**kwargs) for loss, kwargs in config['loss_ftns'].items()]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss_ftns, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--limited_memory', default=False, action='store_true',
                      help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')
    args.add_argument('--representation_model', default='', type=str,
                      help='path to representation model checkpoint (default: None)')
    args.add_argument('--frozen_representation', action="store_true",
                      help='whether to freeze the parameters of representation model')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--rmb', '--reset_monitor_best'], type=bool, target='trainer;reset_monitor_best'),
        CustomArgs(['--vo', '--valid_only'], type=bool, target='trainer;valid_only')
    ]

    representation_model = [args.parse_args().representation_model, args.parse_args().frozen_representation]
    config = ConfigParser.from_args(args, options)

    args = args.parse_args()
    if args.limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        print('---------------multiprocessing---------------')
        torch.multiprocessing.set_sharing_strategy('file_system')
    main(config, args)
