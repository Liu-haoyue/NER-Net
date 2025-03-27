import torch
import model.model as model_arch


def load_model(checkpoint_path, device, sensor_resolution):
    print('loading model from: ', checkpoint_path)
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    model = config.init_obj('arch', model_arch, sensor_resolution)
    # print(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
