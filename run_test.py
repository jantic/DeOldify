from deoldify import device
from deoldify.device_id import DeviceId

from deoldify.visualize import *
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
colorizer = get_image_colorizer(artistic=True)

import os

render_factor=35
#NOTE:  Make source_url None to just read from file at ./video/source/[file_name] directly without modification
source_url=None
result_path = '.'

# change path for source image here
files = os.listdir("/home/v-yuanqidu/old_photo/data/pascal_clean_more_val/train/")
files = ['/home/v-yuanqidu/old_photo/data/pascal_clean_more_val/train/'+f for f in files]
files.extend(['/home/v-yuanqidu/old_photo/data/pascal_clean_more_val/val'+f for f in os.listdir("/home/v-yuanqidu/old_photo/data/pascal_clean_more_val/val/")])

for source_path in files:
    if source_url is not None:
        result_path = colorizer.plot_transformed_image_from_url(url=source_url, path=source_path, render_factor=render_factor, compare=True)
    else:
        result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=True)

show_image_in_notebook(result_path)