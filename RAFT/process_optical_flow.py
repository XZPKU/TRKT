import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import json

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,name_1,name_2,store_path):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = flo
    #img_flo = np.concatenate([img, flo], axis=0)
    out = Image.fromarray((img_flo[:,:,[2,1,0]]).astype(np.uint8))
    out_path = os.path.join(store_path,f'{name_1[-10:-4]}_{name_2[-10:-4]}.png')
    out.save(out_path)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
def viz_origin_flow(img, flo,name_1,name_2,store_path):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    #flo = flow_viz.flow_to_image(flo)
    #img_flo = flo
    #img_flo = np.concatenate([img, flo], axis=0)
    #out = Image.fromarray((img_flo[:,:,[2,1,0]]).astype(np.uint8))
    out_path = os.path.join(store_path,f'{name_1[-10:-4]}_{name_2[-10:-4]}.json')
    with open(out_path,'w') as f:
        json.dump(flo.tolist(),f)
    #out.save(out_path)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    data_root = '/network_space/server127/shared/vidvrd/action-genome/frames'
    store_path = '/network_space/server127/shared/vidvrd/raft_results'
    videos = sorted(os.listdir(data_root))[::-1]
    for video in videos:
        if video.endswith('.mp4'):
            video_root = os.path.join(data_root,video)
            os.makedirs(os.path.join(store_path,video),exist_ok=True)
            try:
                with torch.no_grad():
                    images = glob.glob(os.path.join(video_root, '*.png'))                 
                    images = sorted(images)
                    for imfile1, imfile2 in zip(images, images[1:]):
                        image1 = load_image(imfile1)
                        image2 = load_image(imfile2)

                        padder = InputPadder(image1.shape)
                        image1, image2 = padder.pad(image1, image2)

                        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                        #viz(image1, flow_up,imfile1,imfile2,os.path.join(store_path,video))
                        if os.path.exists(os.path.join(store_path,f'{imfile1[-10:-4]}_{imfile2[-10:-4]}.json')):
                            continue
                        viz_origin_flow(image1, flow_up,imfile1,imfile2,os.path.join(store_path,video))
                print(f'finish {video}')
            except:
                print(f'skip {video}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",default='models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation",default='/network_space/server127/shared/vidvrd/action-genome/frames/0A8CF.mp4')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
