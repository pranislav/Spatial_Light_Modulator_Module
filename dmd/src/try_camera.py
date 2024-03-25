import pylablib as pll
from pylablib.devices import uc480

pll.par["devices/dlls/ueye"] = "/usr/lib/"  # might also be a goot idea to provide a full absolute path just in case
print(uc480.list_cameras(backend="ueye"))