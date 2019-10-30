import os

import torch


class _Device:
    def __init__(self):
        self._current_device = None

    def is_cuda(self):
        ''' Returns `True` if the current device is GPU, `False` otherwise. '''
        return self.current().type == 'cuda'

    def current(self):
        ''' Returns current device. '''
        if self._current_device is None:
            # We could raise an error, but let's default to GPU.
            self.set('cuda')

        return self._current_device

    def set(self, device):
        ''' Set current device. '''
        assert isinstance(device, str), 'device must be str'

        self._current_device = torch.device(device)

        if self.is_cuda():
            if not torch.cuda.is_available():
                raise RuntimeError(
                    'CUDA is not available. Check your system configuration.'
                )

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self._current_device.index)
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.benchmark = False

            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']

        return self._current_device
