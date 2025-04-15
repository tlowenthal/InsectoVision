import os

def search_for_upwards_offset(filename):
    up_levels_path = ""
    i = 0
    while i <= 5 and not os.path.exists(os.path.join(up_levels_path, filename)):
        up_levels_path = os.path.join(up_levels_path, "..")
        i += 1
    return i if i < 6 else -1