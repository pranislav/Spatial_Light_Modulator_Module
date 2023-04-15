from PIL import Image as im

w = 1024
h = 768

def circle(diam: float) -> im:
    target_img = im.new('L', (w, h), 0)
    for i in range(w):
        for j in range(h):
            if (i - w/2)**2 + (j - h/2)**2 < diam**2: #50<i<150 or 50<j<150: #
                target_img.putpixel((i, j), 255)
    return target_img

my_circle = circle(200)