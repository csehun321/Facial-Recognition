import os


def check_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return dir_name
