from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
import random
from utils.image import flip, color_aug,angle2class,class2angle
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

class DddDataset(data.Dataset):

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _bbs_box_to_bbox(self, box):
    bbox = np.array([box.x1, box.y1, box.x2, box.y2],
                    dtype=np.float32)
    return bbox

  def _convert_alpha(self, alpha):
    return math.radians(alpha) if self.alpha_in_degree else alpha

  def __getitem__(self, index):
    scale_range = (0.4, 0.6)
    output_size = (512,512)

    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    img_annos = self.coco.loadAnns(ids=ann_ids)
    annos = [self._coco_box_to_bbox(ann['bbox']) for ann in img_annos]

    num_objs = min(len(annos), self.max_objs)

    aug = iaa.Sequential([
              iaa.AddToHueAndSaturation((-40, 30), per_channel=True),
              iaa.AverageBlur(k=(1, 8)),
          ])

    img = aug(image=img)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    s = np.array([width, height], dtype=np.int32)
    

    trans_input = get_affine_transform(
      c, s, 0, [self.opt.input_w, self.opt.input_h])

    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    num_classes = self.opt.num_classes
    trans_output = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])

    # ===========shape============
    hm = np.zeros(
      (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)

    wh = np.zeros((self.max_objs, 2), dtype=np.float32)

    dep = np.zeros((self.max_objs, 1), dtype=np.float32)

    rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
    rotres = np.zeros((self.max_objs, 2), dtype=np.float32)

    roty = np.zeros((self.max_objs, 1), dtype=np.float32)
    rotx = np.zeros((self.max_objs, 1), dtype=np.float32)
    
    heading_binX = np.zeros((self.max_objs, 1), dtype=np.int64)
    heading_resX = np.zeros((self.max_objs, 1), dtype=np.float32)
    heading_binY = np.zeros((self.max_objs, 1), dtype=np.int64)
    heading_resY = np.zeros((self.max_objs, 1), dtype=np.float32)

    dim = np.zeros((self.max_objs, 3), dtype=np.float32)

    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

    # ===========shape============

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    
    gt_det = []

    for k in range(num_objs):
      ann = img_annos[k]
      bbox = np.array(annos[k])

      cls_id = int(self.cat_ids[1])
      if cls_id <= -99:
        continue

      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)

      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        
        cx_3d = ann['cx']
        cy_3d = ann['cy']

        ct = np.array(
          [cx_3d,cy_3d], dtype=np.float32)/self.opt.down_ratio
        # ct2 = np.array(
        #   [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        # print(ct,ct2)

        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct, radius)

        wh[k] = 1. * w, 1. * h
        gt_det.append([ct_int[1], ct_int[0]])

        alphaX = ann['alphax'] % (2*np.pi)
        alphaY = ann['alphay'] % (2*np.pi)

        heading_binX[k], heading_resX[k] = angle2class(alphaX)
        heading_binY[k], heading_resY[k] = angle2class(alphaY)

        # BINNING
        # bot_thr = np.radians(30)
        # up_thr = np.radians(150)

        # NO OVERLAPPING
        bin_center = np.pi

        if alphaX < bin_center and alphaX > 0:
            bin_class = 0
            rotbin[k, bin_class] = 1
            rotres[k, bin_class] = alphaX 

        if alphaX < 2*bin_center and alphaX > bin_center:
            bin_class = 1
            rotbin[k, bin_class] = 1
            rotres[k, bin_class] = alphaX 

        dep[k] = ann['depth']
        dim[k] = ann['dim']
        roty[k] = alphaY
        rotx[k] = alphaX

        ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1 
        rot_mask[k] = 1

    ret = {'input': inp, 
            'hm': hm, 
            'dep': dep, 
            'dim': dim, 
            'ind': ind,
            'roty':roty,
            'rotx':rotx,
            'rotbin': rotbin, 
            'rotres': rotres,
            'heading_binX': heading_binX, 
            'heading_resX': heading_resX,
            'heading_binY': heading_binY, 
            'heading_resY': heading_resY,
            'reg_mask': reg_mask,
            'rot_mask': rot_mask}

    if self.opt.reg_bbox:
      ret.update({'wh': wh})
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not ('train' in self.split):
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 18), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': 0,
              'image_path': 0, 'img_id': 0,'annos':annos}
      ret['meta'] = meta
    
    return ret

  def _alpha_to_8(self, alpha):
    # return [alpha, 0, 0, 0, 0, 0, 0, 0]
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
