import sys
from IPython.lib import passwd
password = passwd(sys.argv[1])
print(password)
