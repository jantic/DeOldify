import torch
import os


def device(type='gpu'):

    if type.lower() == 'gpu':
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES']='0'
            torch.backends.cudnn.benchmark=True
            torch.device('cuda')
            use_gpu = True
        else:
            raise Exception(f'No GPU found, device = {type}. To use CPU, set device = cpu')
    elif type.lower() == 'cpu':
        torch.device('cpu')
        use_gpu = False
    else:
        raise Exception(f'Set device = gpu or device = cpu')

    device.use_gpu = use_gpu

    print('Using device:', type)
    return
