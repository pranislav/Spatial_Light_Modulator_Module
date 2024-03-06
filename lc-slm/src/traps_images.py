from create_list_of_positions import create_list_of_position_lists, two_circulating_dots_quarterized
from multi_decline_lib import create_sequence_dots

number_of_frames = 360

list_of_position_lists = create_list_of_position_lists(two_circulating_dots_quarterized, number_of_frames, 1)
# print(list_of_position_lists)

create_sequence_dots(list_of_position_lists, "two_circulating_traps_quarterize", "1px")
