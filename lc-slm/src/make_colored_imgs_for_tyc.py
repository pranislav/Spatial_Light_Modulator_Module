import helping_functions_for_slm_generate_etc as hf

source_dir = "holograms/pre_tyca"
hf.process_images_in_dir("holograms/pre_tyca", "holograms/pre_tyca", hf.clut_coloring, hf.make_clut("images/my_clut.png"))
