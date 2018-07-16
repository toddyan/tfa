import sys
import scipy.misc as misc
import PIL.Image as Image
from urllib.request import urlopen
from io import BytesIO
import numpy as np


def image_url_to_nparray(url):
    try:
        response = urlopen(url)
        if response.code == 200:
            img = Image.open(BytesIO(response.read())).convert('RGB').resize((299,299),Image.ANTIALIAS)
            metrix = np.asarray([misc.fromimage(img)])
            return True,metrix
        else:
            print("Http responce not 200")
            return False,None
    except:
        print(sys.exc_info()[0])
        return False,None

if __name__ == "__main__":
    url = "http://puui.qpic.cn/qqvideo_ori/0/g0717yfvkuk_1280_720/0"
    succ, m = image_url_to_nparray(url)
    if succ:
        print(m.shape)
    else:
        print("failed")

