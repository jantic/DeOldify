from fastai.transforms import Transform, TfmType
import cv2

class BlackAndWhiteTransform(Transform):
    def __init__(self, tfm_y=TfmType.NO):
        # Blur strength must be an odd number, because it is used as a kernel size.
        super().__init__(tfm_y)

    def do_transform(self, x, is_y):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)#Gets the 3 channels back
        return x