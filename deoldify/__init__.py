import sys
import logging
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S %z',
                    level=logging.INFO)

from deoldify._device import _Device

device = _Device()
