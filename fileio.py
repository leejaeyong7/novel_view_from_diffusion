import torch
import numpy as np
import re
from struct import pack, unpack
import sys        
from PIL import Image
import torchvision.transforms.functional as F
import json
def read_pfm(pfm_file_path: str)-> torch.Tensor:
    """parses PFM file into torch float tensor

    :param pfm_file_path: path like object that contains full path to the PFM file

    :returns: parsed PFM file of shape CxHxW
    """
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    with open(pfm_file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.ascontiguousarray(np.flip(data, 0))
    return torch.from_numpy(data).view(height, width, -1).permute(2, 0, 1)

def write_pfm(fp, image, scale=1):
    """writes torch.floatarray into PFM file

    :param fp: path like string to write file to
    :param image: numpy binary image that should be of shape HxWx3 or HxW
    :param scale: little / big endian based scale
    """
    color = None
    image = image.detach().cpu().numpy()
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    with open(fp, 'wb') as f:
        header = 'PF\n' if color else 'Pf\n'
        shape = '%d %d\n' % (image.shape[1], image.shape[0])
        f.write(header.encode('utf-8'))
        f.write(shape.encode('utf-8'))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        scale = '%f\n' % scale
        f.write(scale.encode('utf-8'))

        image_string = image.tostring()
        f.write(image_string)