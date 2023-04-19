import imageio
import os

def create_gif(img_dir, outgif_path):
    with imageio.get_writer(outgif_path, mode='I') as writer:
        for file in os.listdir(img_dir):
            image = imageio.imread(f"{img_dir}/{file}")
            writer.append_data(image)
