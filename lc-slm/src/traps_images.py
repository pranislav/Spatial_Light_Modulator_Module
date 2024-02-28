from create_list_of_positions import create_list_of_position_lists, my_parametrization
from multi_decline_lib import create_sequence_dots, multi_decline_img, ellipse

number_of_frames = 10

list_of_position_lists = create_list_of_position_lists(my_parametrization, number_of_frames)
# print(list_of_position_lists)

create_sequence_dots(list_of_position_lists, "moving_traps_4")
