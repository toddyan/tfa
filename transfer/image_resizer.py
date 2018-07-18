import PIL.Image as Image
import os

def image_resize(src, dst):
    img = Image.open(src).convert('RGB').resize((299, 299), Image.ANTIALIAS)
    img.save(dst)

def image_dir_resize(src, dst):
    if os.path.isfile(src):
        Image.open(src).convert('RGB').resize((299, 299), Image.ANTIALIAS).save(dst)

image_resize("D:/tfroot/transfer/1.jpg","D:/tfroot/transfer/299_299/1.jpg")
print(os.path.isdir("D:/tfroot/transfer"))