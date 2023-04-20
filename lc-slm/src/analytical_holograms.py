import slm_screen as ss
import PIL.Image as im
import constants as c

hologram = ss.Screen()

hologram.decline("x", 2 * c.u)

hologram.img.save("holograms/analytical_decline_x2u_y0u.jpg")
