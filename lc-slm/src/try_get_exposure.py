from pylablib.devices import uc480


cam = uc480.UC480Camera()
cam.set_exposure(10e-3)
cam.set_exposure(cam.get_exposure() * 1.1)
print(cam.get_exposure())
