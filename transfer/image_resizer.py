import globalconf
import PIL.Image as Image
import os

def image_resize(src, dst):
    img = Image.open(src).convert('RGB').resize((299, 299), Image.ANTIALIAS)
    img.save(dst)

def image_dir_resize(src, dst):
    if os.path.isfile(src) and src.endswith(".jpg"):
        Image.open(src).convert('RGB').resize((299, 299), Image.ANTIALIAS).save(dst)
    if os.path.isdir(src):
        if not os.path.exists(dst): os.mkdir(dst)
        for f in os.listdir(src):
            image_dir_resize(os.path.join(src, f),os.path.join(dst, f))

if __name__ == "__main__":
    src = globalconf.get_root() + "transfer/flower_photos"
    dst = globalconf.get_root() + "transfer/small"
    image_dir_resize(src, dst)