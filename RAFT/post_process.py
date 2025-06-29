import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import os
import json    
import glob
import numpy as np
from tqdm import tqdm

with open('/network_space/server127/shared/vidvrd/action-genome/weak_ag_det_coco_style_with_det_test.json') as f:
    target_info = json.load(f)
target_images = target_info['images']
target_anno = target_info['annotations']
target_categories = target_info['categories'] 
target_rel_categories = target_info['rel_categories']
target_detections = target_info['detections']
raft_info = []
raft_result_path = '/network_space/server127/shared/vidvrd/raft_results'
image_path  ='/network_space/server127/shared/vidvrd/action-genome/frames'
for item in tqdm(target_images):
    image_name = item['file_name'][7:16]
    raft_path = os.path.join(raft_result_path,image_name)
    img_idx = item['file_name'][17:23]
    files = glob.glob(os.path.join(raft_path,f'*_{img_idx}.json'))
    if len(files) > 0:
        try:
            file = files[0]
            with open(file,'r') as f:
                raft_info = json.load(f)
            previous_idx = file[-18:-12]
            previous_path  =os.path.join(image_path,f'{image_name}/{previous_idx}.png')
            previous_info = np.array(Image.open(previous_path))
            image_id = item['id']
        except:
            current_image = np.array(Image.open(os.path.join(image_path,f'{image_name}/{img_idx}.png')))
            h, w = current_image.shape[:2]
            raft_info = list(np.zeros((h, w, 2)))
            previous_info = current_image
            image_id = item['id']
    else:
        current_image = np.array(Image.open(os.path.join(image_path,f'{image_name}/{img_idx}.png')))
        h, w = current_image.shape[:2]
        raft_info = list(np.zeros((h, w, 2)))
        previous_info = current_image
        image_id = item['id']
        
    raft_info_for_each_image = {}
    raft_info_for_each_image['image_id'] = image_id
    raft_info_for_each_image['raft'] = raft_info
    raft_info_for_each_image['previous_image'] = previous_info
    raft_info.append(raft_info_for_each_image)
final_info = {}
final_info['images'] = target_images
final_info['annotations'] = target_anno
final_info['categories'] = target_categories
final_info['rel_categories'] = target_rel_categories
final_info['detections'] = target_detections
final_info['raft_info'] = raft_info
with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft.json','w') as f:
    json.dump(final_info,f)