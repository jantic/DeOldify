from deoldify import device
from deoldify.device_id import DeviceId

device.set(device=DeviceId.GPU0)

import fastai
import torch
import warnings
import pathlib

from deoldify.visualize import *

from cog import BasePredictor, Path, Input

torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    print("GPU not available.")


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "-p", "/root/.cache/torch/hub/checkpoints"])
        subprocess.run(
            [
                "cp",
                "-r",
                "resnet34-b627a593.pth",
                "/root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth",
            ]
        )
        subprocess.run(
            [
                "cp",
                "-r",
                "resnet101-63fe2227.pth",
                "/root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth",
            ]
        )

    def predict(
        self,
        input_image: Path = Input(description="Path to an image"),
        model_name: str = Input(
            description="Which model to use: "
            "Artistic has more vibrant color but may leave important parts of the image gray."
            "Stable is better for nature scenery and is less prone to leaving gray human parts",
            choices=["Artistic", "Stable"],
        ),
        render_factor: int = Input(
            description="The default value of 35 has been carefully chosen and should work "
            "-ok- for most scenarios (but probably won't be the -best-"
            "). This determines resolution at which the color portion of "
            "the image is rendered. Lower resolution will render faster, "
            "and colors also tend to look more vibrant. Older and lower quality "
            "images in particular will generally benefit by lowering the render "
            "factor. Higher render factors are often better for higher quality "
            "images, but the colors may get slightly washed out.",
            default=35,
        ),
    ) -> Path:
        input_image = str(input_image)
        colorizer = get_image_colorizer(
            artistic=str(model_name) == "Artistic"
        )  # TODO: check if this have to be in setup

        output_path = colorizer.plot_transformed_image(
            path=input_image,
            results_dir=pathlib.Path("."),
            render_factor=int(render_factor),
            compare=True,
            watermarked=True,
        )

        return Path(output_path)
