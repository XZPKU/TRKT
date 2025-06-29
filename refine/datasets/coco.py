# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import torch.nn.functional as F
import pdb
import cv2
import numpy as np
from PIL import Image
import datasets.transforms as T
import torchvision.transforms as transforms
from .coco_torchvision import CocoDetection
import torchvision.transforms as tensor_transforms
def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset

class CocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset='coco', full_ratio=0.0):
        real_full_ratio = full_ratio if 'train' in str(img_folder) else 0.0
        super(CocoDetection, self).__init__(img_folder, ann_file, full_ratio=full_ratio)
        self._transforms = transforms
        self.dataset = dataset
        self.withDet = 'detections' in self.coco.dataset
        self.prepare = ConvertCocoPolysToMask(return_masks, dataset=self.dataset)
        
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        #print(self.withDet)
        if self.withDet:
            img, target, detection = super(CocoDetection, self).__getitem__(idx)
            target = {'image_id': image_id, 'annotations': target, 'detections': detection}
        else:
            img, target = super(CocoDetection, self).__getitem__(idx)
            target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #print(target)
        return img, target

class CocoDetection_w_raft(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset='coco', full_ratio=0.0):
        real_full_ratio = full_ratio if 'train' in str(img_folder) else 0.0
        super(CocoDetection, self).__init__(img_folder, ann_file, full_ratio=full_ratio)
        self._transforms = transforms
        self.dataset = dataset
        self.withDet = 'detections' in self.coco.dataset
        self.prepare = ConvertCocoPolysToMask_w_raft(return_masks, dataset=self.dataset)
        if 'train' in str(ann_file):
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_train_add_raft_k_v.json','r') as f:
                self.optical = json.load(f)
            # with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_train_add_raft_k_v_next.json','r') as f:
            #     self.optical_next = json.load(f)
        else:
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft_k_v.json','r') as f:
                self.optical = json.load(f)
            # with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft_k_v_next.json','r') as f:
            #     self.optical_next = json.load(f)
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        #print(self.withDet)
        if self.withDet:
            img, target, detection = super(CocoDetection, self).__getitem__(idx)
            #print(f'current shape is {img.size}')
            # import pdb
            # pdb.set_trace()
            origin_path = self.coco.loadImgs(image_id)[0]["file_name"]
            #print(origin_path)
            
            ############################  prrevious frame ###########################
            optical_info  =self.optical[origin_path]
            raft_path  =optical_info['raft']
            previous_image_path = optical_info['previous_image']
            #print(optical_info)
            previous_image = Image.open(previous_image_path).convert('RGB')
            previous_image, _ = self._transforms(previous_image,None)
            #print(f'previous shape is {previous_image.size}')
            
            if raft_path is not None:
                try:
                    with open(raft_path,'r') as f:
                        raft = json.load(f)
                    #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
                    raft = torch.tensor(raft) # h*w*2
                    flow = raft.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
                    resized_flow = F.interpolate(flow, size=(32, 32), mode='bilinear', align_corners=False)

                    # 变回原始维度顺序 (H, W, C)
                    raft = resized_flow.squeeze(0).permute(1,2,0) # 2*h*w
                except:
                    raft = torch.zeros([32,32,2])
                    #previous_image = img
            else:
                # raft = torch.zeros([img.size[0],img.size[1],2])
                raft = torch.zeros([32,32,2])
            
            ############################  next frame ###########################
            
            # optical_info_next  =self.optical_next[origin_path]
            # raft_path_next  =optical_info_next['raft']
            # next_image_path = optical_info_next['next_image']
            # #print(optical_info)
            # next_image = Image.open(next_image_path).convert('RGB')
            # next_image, _ = self._transforms(next_image,None)
            # #print(f'previous shape is {previous_image.size}')
            
            # if raft_path_next is not None:
            #     try:
            #         with open(raft_path_next,'r') as f:
            #             raft_next = json.load(f)
            #         #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
            #         raft_next = torch.tensor(raft_next) # h*w*2
            #         flow_next = raft_next.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
            #         resized_flow_next = F.interpolate(flow_next, size=(32, 32), mode='bilinear', align_corners=False)

            #         # 变回原始维度顺序 (H, W, C)
            #         raft_next = resized_flow_next.squeeze(0).permute(1,2,0) # 2*h*w
            #     except:
            #         raft_next = torch.zeros([32,32,2])
            #         #next_image  = img
            # else:
            #     # raft = torch.zeros([img.size[0],img.size[1],2])
            #     raft_next = torch.zeros([32,32,2])
                
                
            #print(f'raft shape is {raft.shape}')
            target = {'image_id': image_id, 'annotations': target, 'detections': detection,'raft': raft, 'previous': previous_image}
            #print(target.keys())
        else:
            img, target = super(CocoDetection, self).__getitem__(idx)
            target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #print(img.shape)
        return img, target

class CocoDetection_w_raft_exclude_leak(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset='coco', full_ratio=0.0):
        real_full_ratio = full_ratio if 'train' in str(img_folder) else 0.0
        super(CocoDetection, self).__init__(img_folder, ann_file, full_ratio=full_ratio)
        self._transforms = transforms
        self.dataset = dataset
        self.withDet = 'detections' in self.coco.dataset
        self.prepare = ConvertCocoPolysToMask_w_raft(return_masks, dataset=self.dataset)
        if 'train' in str(ann_file):
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_train_add_raft_k_v.json','r') as f:
                self.optical = json.load(f)
            with open('/network_space/server126/shared/xuzhu/CSA/refine/train_frames.json','r') as f:
                self.valid = json.load(f)
            # with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_train_add_raft_k_v_next.json','r') as f:
            #     self.optical_next = json.load(f)
        else:
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft_k_v.json','r') as f:
                self.optical = json.load(f)
            with open('/network_space/server126/shared/xuzhu/CSA/refine/test_frames.json','r') as f:
                self.valid = json.load(f)
            # with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft_k_v_next.json','r') as f:
            #     self.optical_next = json.load(f)
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        #print(self.withDet)
        if self.withDet:
            img, target, detection = super(CocoDetection, self).__getitem__(idx)
            #print(f'current shape is {img.size}')
            # import pdb
            # pdb.set_trace()
            origin_path = self.coco.loadImgs(image_id)[0]["file_name"]
            #print(origin_path)
            
            ############################  prrevious frame ###########################
            optical_info  =self.optical[origin_path]
            raft_path  =optical_info['raft']
            previous_image_path = optical_info['previous_image']
            # print(f'previous image {previous_image_path[-27:]}')
            # print(f'self.valid {self.valid[0]}')
            if previous_image_path[-27:] in self.valid:
                #print('valid!')
                #print(optical_info)
                previous_image = Image.open(previous_image_path).convert('RGB')
                previous_image, _ = self._transforms(previous_image,None)
                #print(f'previous shape is {previous_image.size}')
                
                if raft_path is not None:
                    try:
                        with open(raft_path,'r') as f:
                            raft = json.load(f)
                        #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
                        raft = torch.tensor(raft) # h*w*2
                        flow = raft.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
                        resized_flow = F.interpolate(flow, size=(32, 32), mode='bilinear', align_corners=False)

                        # 变回原始维度顺序 (H, W, C)
                        raft = resized_flow.squeeze(0).permute(1,2,0) # 2*h*w
                    except:
                        raft = torch.zeros([32,32,2])
                        #previous_image = img
                else:
                    # raft = torch.zeros([img.size[0],img.size[1],2])
                    raft = torch.zeros([32,32,2])
            else:
                #print('not valid!')
                previous_image, _ = self._transforms(img,None)
                raft = torch.zeros([32,32,2])
            ############################  next frame ###########################
            
            # optical_info_next  =self.optical_next[origin_path]
            # raft_path_next  =optical_info_next['raft']
            # next_image_path = optical_info_next['next_image']
            # #print(optical_info)
            # next_image = Image.open(next_image_path).convert('RGB')
            # next_image, _ = self._transforms(next_image,None)
            # #print(f'previous shape is {previous_image.size}')
            
            # if raft_path_next is not None:
            #     try:
            #         with open(raft_path_next,'r') as f:
            #             raft_next = json.load(f)
            #         #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
            #         raft_next = torch.tensor(raft_next) # h*w*2
            #         flow_next = raft_next.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
            #         resized_flow_next = F.interpolate(flow_next, size=(32, 32), mode='bilinear', align_corners=False)

            #         # 变回原始维度顺序 (H, W, C)
            #         raft_next = resized_flow_next.squeeze(0).permute(1,2,0) # 2*h*w
            #     except:
            #         raft_next = torch.zeros([32,32,2])
            #         #next_image  = img
            # else:
            #     # raft = torch.zeros([img.size[0],img.size[1],2])
            #     raft_next = torch.zeros([32,32,2])
                
                
            #print(f'raft shape is {raft.shape}')
            target = {'image_id': image_id, 'annotations': target, 'detections': detection,'raft': raft, 'previous': previous_image}
            #print(target.keys())
        else:
            img, target = super(CocoDetection, self).__getitem__(idx)
            origin_path = self.coco.loadImgs(image_id)[0]["file_name"]
            #print(origin_path)
            
            ############################  prrevious frame ###########################
            # optical_info  =self.optical[origin_path]
            # raft_path  =optical_info['raft']
            # previous_image_path = optical_info['previous_image']
            # # print(f'previous image {previous_image_path[-27:]}')
            # # print(f'self.valid {self.valid[0]}')
            # if previous_image_path[-27:] in self.valid:
            #     print('valid!')
            #     #print(optical_info)
            #     previous_image = Image.open(previous_image_path).convert('RGB')
            #     previous_image, _ = self._transforms(previous_image,None)
            #     #print(f'previous shape is {previous_image.size}')
                
            #     if raft_path is not None:
            #         try:
            #             with open(raft_path,'r') as f:
            #                 raft = json.load(f)
            #             #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
            #             raft = torch.tensor(raft) # h*w*2
            #             flow = raft.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
            #             resized_flow = F.interpolate(flow, size=(32, 32), mode='bilinear', align_corners=False)

            #             # 变回原始维度顺序 (H, W, C)
            #             raft = resized_flow.squeeze(0).permute(1,2,0) # 2*h*w
            #         except:
            #             raft = torch.zeros([32,32,2])
            #             #previous_image = img
            #     else:
            #         # raft = torch.zeros([img.size[0],img.size[1],2])
            #         raft = torch.zeros([32,32,2])
            # else:
            #print('not valid!')
            previous_image, _ = self._transforms(img,None)
            raft = torch.zeros([32,32,2])
            target = {'image_id': image_id, 'annotations': target, 'raft': raft, 'previous': previous_image}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #print(img.shape)
        return img, target
    
class CocoDetection_w_raft_next(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset='coco', full_ratio=0.0):
        real_full_ratio = full_ratio if 'train' in str(img_folder) else 0.0
        super(CocoDetection, self).__init__(img_folder, ann_file, full_ratio=full_ratio)
        self._transforms = transforms
        self.dataset = dataset
        self.withDet = 'detections' in self.coco.dataset
        self.prepare = ConvertCocoPolysToMask_w_raft(return_masks, dataset=self.dataset)
        if 'train' in str(ann_file):
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_train_add_raft_k_v.json','r') as f:
                self.optical = json.load(f)
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_train_add_raft_k_v_next.json','r') as f:
                self.optical_next = json.load(f)
        else:
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft_k_v.json','r') as f:
                self.optical = json.load(f)
            with open('/network_space/server126/shared/xuzhu/CSA/refine/weak_ag_det_coco_style_with_det_test_add_raft_k_v_next.json','r') as f:
                self.optical_next = json.load(f)
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        #print(self.withDet)
        if self.withDet:
            img, target, detection = super(CocoDetection, self).__getitem__(idx)
            #print(f'current shape is {img.size}')
            # import pdb
            # pdb.set_trace()
            origin_path = self.coco.loadImgs(image_id)[0]["file_name"]
            #print(origin_path)
            
            ############################  prrevious frame ###########################
            optical_info  =self.optical[origin_path]
            raft_path  =optical_info['raft']
            previous_image_path = optical_info['previous_image']
            #print(optical_info)
            previous_image = Image.open(previous_image_path).convert('RGB')
            previous_image, _ = self._transforms(previous_image,None)
            #print(f'previous shape is {previous_image.size}')
            
            if raft_path is not None:
                try:
                    with open(raft_path,'r') as f:
                        raft = json.load(f)
                    #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
                    raft = torch.tensor(raft) # h*w*2
                    flow = raft.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
                    resized_flow = F.interpolate(flow, size=(32, 32), mode='bilinear', align_corners=False)

                    # 变回原始维度顺序 (H, W, C)
                    raft = resized_flow.squeeze(0).permute(1,2,0) # 2*h*w
                except:
                    raft = torch.zeros([32,32,2])
                    #previous_image = img
            else:
                # raft = torch.zeros([img.size[0],img.size[1],2])
                raft = torch.zeros([32,32,2])
            
            ############################  next frame ###########################
            
            optical_info_next  =self.optical_next[origin_path]
            raft_path_next  =optical_info_next['raft']
            next_image_path = optical_info_next['next_image']
            #print(optical_info)
            next_image = Image.open(next_image_path).convert('RGB')
            next_image, _ = self._transforms(next_image,None)
            #print(f'previous shape is {previous_image.size}')
            
            if raft_path_next is not None:
                try:
                    with open(raft_path_next,'r') as f:
                        raft_next = json.load(f)
                    #raft = cv2.resize(raft, (512, 512), interpolation=cv2.INTER_LINEAR)
                    raft_next = torch.tensor(raft_next) # h*w*2
                    flow_next = raft_next.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 2, h, w)
                    resized_flow_next = F.interpolate(flow_next, size=(32, 32), mode='bilinear', align_corners=False)

                    # 变回原始维度顺序 (H, W, C)
                    raft_next = resized_flow_next.squeeze(0).permute(1,2,0) # 2*h*w
                except:
                    raft_next = torch.zeros([32,32,2])
                    #next_image  = img
            else:
                # raft = torch.zeros([img.size[0],img.size[1],2])
                raft_next = torch.zeros([32,32,2])
                
                
            #print(f'raft shape is {raft.shape}')
            target = {'image_id': image_id, 'annotations': target, 'detections': detection,'raft': raft, 'previous': previous_image,'next': next_image, 'raft_next': raft_next}
            #print(target.keys())
        else:
            img, target = super(CocoDetection, self).__getitem__(idx)
            target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #print(img.shape)
        return img, target
    
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, dataset="voc"):
        self.return_masks = return_masks
        self.dataset = dataset
        print(self.dataset)
        if self.dataset == "coco":
            self.num_classes = 36
        else:
            self.num_classes = 20
        self.num_attn_classes = 3
        self.num_spatial_classes = 6
        self.num_contacting_classes = 17

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
 
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        classes_one_hot = F.one_hot(classes-1, self.num_classes).sum(dim=0).clamp(0,1).long()
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        
        attn_one_hot = torch.zeros((len(anno), self.num_attn_classes), dtype=torch.int64)
        spatial_one_hot = torch.zeros((len(anno), self.num_spatial_classes), dtype=torch.int64)
        contacting_one_hot = torch.zeros((len(anno), self.num_contacting_classes), dtype=torch.int64)

        for idx, obj in enumerate(anno):
            if len(obj['attention_rel']) > 0:
                att_idx = torch.tensor(anno[idx]['attention_rel'])
                spa_idx = torch.tensor(anno[idx]['spatial_rel']) - self.num_attn_classes
                con_idx = torch.tensor(anno[idx]['contact_rel']) - self.num_attn_classes - self.num_spatial_classes

                attn_one_hot[idx, att_idx] = 1
                spatial_one_hot[idx, spa_idx] = 1
                contacting_one_hot[idx, con_idx] = 1
        # attn_one_hot[0] = attn_one_hot[1:].sum(dim=0) # TODO: person + all relations

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        det_boxes = None
        det_classes = None
        det_scores = None
        if 'detections' in target:
            det = target['detections']
 
            det = [obj for obj in det]

            det_boxes = [obj["bbox"] for obj in det]
            # guard against no boxes via resizing
            det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32).reshape(-1, 4)
            # det_boxes[:, 2:] += det_boxes[:, :2] # wrong!
            det_boxes[:, 0::2].clamp_(min=0, max=w)
            det_boxes[:, 1::2].clamp_(min=0, max=h)

            det_classes = [obj["category_id"] for obj in det]
            det_classes = torch.tensor(det_classes, dtype=torch.int64)

            det_scores = [obj["score"] for obj in det]
            det_scores = torch.tensor(det_scores, dtype=torch.float32)

            det_keep = (det_boxes[:, 3] > det_boxes[:, 1]) & (det_boxes[:, 2] > det_boxes[:, 0])
            det_boxes = det_boxes[det_keep]
            det_classes = det_classes[det_keep]
            det_scores = det_scores[det_keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["attention_label"] = attn_one_hot
        target["spatial_label"] = spatial_one_hot
        target["contact_label"] = contacting_one_hot
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        target["img_label"] = classes_one_hot

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if det_boxes is not None:
            target["det_boxes"] = det_boxes
            target["det_labels"] = det_classes
            target["det_scores"] = det_scores
        
        target['is_full'] = torch.ones(1) if anno[0]['is_full'] else torch.zeros(1)

        return image, target
class ConvertCocoPolysToMask_w_raft(object):
    def __init__(self, return_masks=False, dataset="voc"):
        self.return_masks = return_masks
        self.dataset = dataset
        print(self.dataset)
        if self.dataset == "coco":
            self.num_classes = 36
        else:
            self.num_classes = 20
        self.num_attn_classes = 3
        self.num_spatial_classes = 6
        self.num_contacting_classes = 17

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
 
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        classes_one_hot = F.one_hot(classes-1, self.num_classes).sum(dim=0).clamp(0,1).long()
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        
        attn_one_hot = torch.zeros((len(anno), self.num_attn_classes), dtype=torch.int64)
        spatial_one_hot = torch.zeros((len(anno), self.num_spatial_classes), dtype=torch.int64)
        contacting_one_hot = torch.zeros((len(anno), self.num_contacting_classes), dtype=torch.int64)

        for idx, obj in enumerate(anno):
            if len(obj['attention_rel']) > 0:
                att_idx = torch.tensor(anno[idx]['attention_rel'])
                spa_idx = torch.tensor(anno[idx]['spatial_rel']) - self.num_attn_classes
                con_idx = torch.tensor(anno[idx]['contact_rel']) - self.num_attn_classes - self.num_spatial_classes

                attn_one_hot[idx, att_idx] = 1
                spatial_one_hot[idx, spa_idx] = 1
                contacting_one_hot[idx, con_idx] = 1
        # attn_one_hot[0] = attn_one_hot[1:].sum(dim=0) # TODO: person + all relations

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        det_boxes = None
        det_classes = None
        det_scores = None
        if 'detections' in target:
            det = target['detections']
 
            det = [obj for obj in det]

            det_boxes = [obj["bbox"] for obj in det]
            # guard against no boxes via resizing
            det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32).reshape(-1, 4)
            # det_boxes[:, 2:] += det_boxes[:, :2] # wrong!
            det_boxes[:, 0::2].clamp_(min=0, max=w)
            det_boxes[:, 1::2].clamp_(min=0, max=h)

            det_classes = [obj["category_id"] for obj in det]
            det_classes = torch.tensor(det_classes, dtype=torch.int64)

            det_scores = [obj["score"] for obj in det]
            det_scores = torch.tensor(det_scores, dtype=torch.float32)

            det_keep = (det_boxes[:, 3] > det_boxes[:, 1]) & (det_boxes[:, 2] > det_boxes[:, 0])
            det_boxes = det_boxes[det_keep]
            det_classes = det_classes[det_keep]
            det_scores = det_scores[det_keep]
        # if 'raft' in target:
        raft = target['raft']
        previous_image = target['previous']
        # raft_next = target['raft_next']
        # next_image = target['next']
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["attention_label"] = attn_one_hot
        target["spatial_label"] = spatial_one_hot
        target["contact_label"] = contacting_one_hot
        target["raft"] = raft
        #trans_func = tensor_transforms.ToTensor()
        #print(f'previous image is {previous_image}, the sahpe is {previous_image.shape}')
        target["previous"] = previous_image
        #target['raft_next'] = raft_next
        #target['next'] = next_image
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        target["img_label"] = classes_one_hot

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if det_boxes is not None:
            target["det_boxes"] = det_boxes
            target["det_labels"] = det_classes
            target["det_scores"] = det_scores
        
        target['is_full'] = torch.ones(1) if anno[0]['is_full'] else torch.zeros(1)

        return image, target

class ConvertCocoPolysToMask_w_raft_next(object):
    def __init__(self, return_masks=False, dataset="voc"):
        self.return_masks = return_masks
        self.dataset = dataset
        print(self.dataset)
        if self.dataset == "coco":
            self.num_classes = 36
        else:
            self.num_classes = 20
        self.num_attn_classes = 3
        self.num_spatial_classes = 6
        self.num_contacting_classes = 17

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
 
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        classes_one_hot = F.one_hot(classes-1, self.num_classes).sum(dim=0).clamp(0,1).long()
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        
        attn_one_hot = torch.zeros((len(anno), self.num_attn_classes), dtype=torch.int64)
        spatial_one_hot = torch.zeros((len(anno), self.num_spatial_classes), dtype=torch.int64)
        contacting_one_hot = torch.zeros((len(anno), self.num_contacting_classes), dtype=torch.int64)

        for idx, obj in enumerate(anno):
            if len(obj['attention_rel']) > 0:
                att_idx = torch.tensor(anno[idx]['attention_rel'])
                spa_idx = torch.tensor(anno[idx]['spatial_rel']) - self.num_attn_classes
                con_idx = torch.tensor(anno[idx]['contact_rel']) - self.num_attn_classes - self.num_spatial_classes

                attn_one_hot[idx, att_idx] = 1
                spatial_one_hot[idx, spa_idx] = 1
                contacting_one_hot[idx, con_idx] = 1
        # attn_one_hot[0] = attn_one_hot[1:].sum(dim=0) # TODO: person + all relations

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        det_boxes = None
        det_classes = None
        det_scores = None
        if 'detections' in target:
            det = target['detections']
 
            det = [obj for obj in det]

            det_boxes = [obj["bbox"] for obj in det]
            # guard against no boxes via resizing
            det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32).reshape(-1, 4)
            # det_boxes[:, 2:] += det_boxes[:, :2] # wrong!
            det_boxes[:, 0::2].clamp_(min=0, max=w)
            det_boxes[:, 1::2].clamp_(min=0, max=h)

            det_classes = [obj["category_id"] for obj in det]
            det_classes = torch.tensor(det_classes, dtype=torch.int64)

            det_scores = [obj["score"] for obj in det]
            det_scores = torch.tensor(det_scores, dtype=torch.float32)

            det_keep = (det_boxes[:, 3] > det_boxes[:, 1]) & (det_boxes[:, 2] > det_boxes[:, 0])
            det_boxes = det_boxes[det_keep]
            det_classes = det_classes[det_keep]
            det_scores = det_scores[det_keep]
        # if 'raft' in target:
        raft = target['raft']
        previous_image = target['previous']
        raft_next = target['raft_next']
        next_image = target['next']
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["attention_label"] = attn_one_hot
        target["spatial_label"] = spatial_one_hot
        target["contact_label"] = contacting_one_hot
        target["raft"] = raft
        #trans_func = tensor_transforms.ToTensor()
        #print(f'previous image is {previous_image}, the sahpe is {previous_image.shape}')
        target["previous"] = previous_image
        target['raft_next'] = raft_next
        target['next'] = next_image
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        target["img_label"] = classes_one_hot

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if det_boxes is not None:
            target["det_boxes"] = det_boxes
            target["det_labels"] = det_classes
            target["det_scores"] = det_scores
        
        target['is_full'] = torch.ones(1) if anno[0]['is_full'] else torch.zeros(1)

        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_specific_size(image_set, max_size=1333):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales_old = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    Random_size = [400, 500, 600]
    Crop_size = [384, 600]
    Random_size = [(r * max_size // 1333) for r in Random_size] 
    Crop_size = [(c * max_size // 1333) for c in Crop_size] 
    scales = []
    
    for s in scales_old:
        scales.append(s * max_size // 1333)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(Random_size),
                    T.RandomSizeCrop(*Crop_size),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.Resize((max_size,max_size), max_size=max_size),
            T.RandomResize([800*max_size//1333], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_specific_size_fixed(image_set, max_size=1333):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales_old = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    Random_size = [384, 448, 512]
    Crop_size = [384, 600]
    Crop_size = [(c * max_size // 1333) for c in Crop_size]
    scales = []
    for s in scales_old:
        scales.append(s * max_size // 1333)
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize((max_size,max_size), max_size=max_size),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize((max_size, max_size), max_size=max_size),
            # T.RandomResize([800*max_size//1333], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided VOC path {root} does not exist'
    mode = 'instances'
    if args.ext_det is False:
        pdb.set_trace()
        PATHS = {
            "train": (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_train.json"), 
            "val": (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_test.json")
        }
    else:
        PATHS = {
            "train": (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_with_det_train.json"), 
            "val": (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_with_det_test.json")
        }
        # ############## alter to dino detection results ######################3
        # PATHS = {
        #     "train": (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_with_dino_det_train_v2.json"), 
        #     "val": (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_with_dino_det_test_v2.json")
        # }
        if args.infer_save:
            if args.infer_set == 'train':
                PATHS['val'] = (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_with_det_train.json")
            elif args.infer_set == 'test':
                PATHS['val'] = (root, root / "annotations" / "weak" / "weak_ag_det_coco_style_with_det_test.json")
  
            elif args.infer_set == 'total':
                PATHS['val'] = (root, root / "annotations" / "ag_train_coco_style.json")

    img_folder, ann_file = PATHS[image_set]
    if args.fixed_size:
        #dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_specific_size_fixed(image_set, max_size=args.max_size), return_masks=False, dataset='coco', full_ratio=args.full_ratio)
        #dataset = CocoDetection_w_raft(img_folder, ann_file, transforms=make_coco_transforms_specific_size_fixed(image_set, max_size=512), return_masks=False, dataset='coco', full_ratio=0.0)
        dataset = CocoDetection_w_raft_exclude_leak(img_folder, ann_file, transforms=make_coco_transforms_specific_size_fixed(image_set, max_size=512), return_masks=False, dataset='coco', full_ratio=0.0)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_specific_size(image_set, max_size=args.max_size), return_masks=False, dataset='coco', full_ratio=args.full_ratio)
    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)
    return dataset

def build_voc_psuedo(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided VOC path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images", "/home/zhangjiahua/Workspace/weak_det/TransLocVOC3/data/voc_0712_psuedo_coco/voc_0712_trainval.json"),
        "val": (root / "images", root / "annotations" / "voc_2007_test.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_specific_size(image_set, max_size=400), return_masks=False)
    
    return dataset

