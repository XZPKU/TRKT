from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import torch.nn.functional as F
import pdb
from coco import build


parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--fixed_size', action='store_false')
parser.add_argument('--coco_path', default='/network_space/server127/shared/vidvrd/action-genome', type=float)
parser.add_argument('--infer_save',action='store_true')
parser.add_argument('--ext_det', action='store_false')
args = parser.parse_args()
data = build(image_set='train',args = args)