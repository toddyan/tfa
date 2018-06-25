import platform


def get_root():
    if platform.system() == "Windows":
        return "D:/tfroot/"
    elif platform.system() == "Linux":
        return "/home/yxd/tfroot/"
    elif platform.system() == "Darwin":
        return "/Users/yxd/tfroot/"
    return ""
