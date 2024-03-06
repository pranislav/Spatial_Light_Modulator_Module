import slm_screen as ss
import PIL.Image as im
import constants as c

hologram = ss.Screen()

hologram.decline("x", 2 * c.u).decline("y", 2 * c.u)

hologram.img.save("holograms/analytical_decline_x2u_y2u.jpg")

# hologram.lens(0.6)
# hologram.img.save("holograms/analytical_xdecline_2u.jpg")