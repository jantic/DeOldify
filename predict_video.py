from deoldify import device
from deoldify.device_id import DeviceId

device.set(device=DeviceId.GPU0)

import fastai
import torch
import warnings
import pathlib

from deoldify.visualize import *

torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    print("GPU not available.")

from cog import BasePredictor, Path, Input

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*?Your .*? set is empty.*?"
)


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "-p", "/root/.cache/torch/hub/checkpoints"])
        subprocess.run(
            [
                "cp",
                "-r",
                "resnet101-63fe2227.pth",
                "/root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth",
            ]
        )
        self.colorizer = get_video_colorizer()

    def predict(
        self,
        input_video: Path = Input(description="Path to a video"),
        render_factor: int = Input(
            description="The default value of 35 has been carefully chosen and should work "
            "-ok- for most scenarios (but probably won't be the -best-"
            "). This determines resolution at which the color portion of "
            "the image is rendered. Lower resolution will render faster, "
            "and colors also tend to look more vibrant. Older and lower quality "
            "images in particular will generally benefit by lowering the render "
            "factor. Higher render factors are often better for higher quality "
            "images, but the colors may get slightly washed out.",
            default=21,
        ),
    ) -> Path:
        input_video = str(input_video)

        output_path = self.colorizer.colorize_from_file_name(
            file_name=input_video,
            render_factor=int(render_factor),
            watermarked=True,
        )

        return Path(output_path)
