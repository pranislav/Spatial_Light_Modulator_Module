import slm_screen as sc
import hologram_GD_2D as gd

u = sc.first_diff_max/4

img_name = "apple"
img_type = "jpg"
algorithm = "GD" # GD or GS


def generate_hologram(image, algorithm):
    # daco na sposob slm.gs, ale unifikovat aj s GD
    pass

def transform_hologram(hologram, angle, focal_len):
    if angle:
        x_angle, y_angle = angle
        return sc.Screen(hologram).decline('x', x_angle).decline('y', y_angle)
    if focal_len:
        return sc.Screen(hologram).lens(focal_len)
    else:
        return sc.Screen(hologram)

# screen = sc.Screen(slm_gs.source_phase).decline('x', sc.first_diff_max/4).decline('y', sc.first_diff_max/4)

# screen.img.show()
# screen.img.convert("RGB").save(f"holograms/{slm_gs.target_name}_hologram.jpg")


hologram = generate_hologram(f"images/{img_name}.{img_type}", algorithm)
hologram_sc = transform_hologram(hologram, (1*u, 1*u), False)
hologram_sc.img.show()
hologram_sc.img.convert("RGB").save(f"holograms/{img_name}_hologram.jpg")
